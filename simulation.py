import os
import numpy as np
import dolfin as df
import argparse
import random
import math
from block import block_assemble, block_mat, block_vec
from block.algebraic.petsc import LU 
from block.iterative import MinRes 

from niitumorio import load_nii, create_path
from dolfintumorio import load_field 


def load_2d_slice():
    """Loads all the data needed to run a simulation."""
    directory = 'data-bwohlmuth'
    patient_id = '008'
    timepoint = 'postop'

    slice_z = 79
    offsets_left = [40,20,slice_z]
    offsets_right = [-40,-20,slice_z+1]

    pixel_scale=1e-3

    path_bmask = create_path(directory, patient_id, timepoint, data='brainmask')
    matter = load_nii(path_bmask, offsets_left, offsets_right)

    print(matter.shape)
    Nx, Ny = (np.array(matter.shape) / 1).astype(np.int32)
    mesh = df.RectangleMesh(df.Point(0,0), df.Point(Nx*pixel_scale, Ny*pixel_scale), Nx, Ny)
    DG0 = df.FunctionSpace(mesh, 'DG', 0)

    # load the tissue:
    matter_path = create_path(directory, patient_id, timepoint, data='tissuemask')
    matter = load_nii(matter_path, offsets_left, offsets_right)
    # white matter
    np_wm = np.zeros(matter.shape) 
    np_wm[matter == 3] = 1.
    wm_field = load_field(DG0, np_wm, 'wm', pixel_scale)
    # gray matter
    np_gm = np.zeros(matter.shape) 
    np_gm[matter == 2] = 1.
    gm_field = load_field(DG0, np_gm, 'gm', pixel_scale)
    # cerespinal fluid 
    np_csf = np.zeros(matter.shape) 
    np_csf[matter == 1] = 1.
    csf_field = load_field(DG0, np_csf, 'csf', pixel_scale)

    # load the segmentation and use it for initial conditions:
    segmentation_path = create_path(directory, patient_id, timepoint, data='seg')
    np_seg = load_nii(segmentation_path, offsets_left, offsets_right)
    # set endema to something small
    np_seg[2 == np_seg] = 0.05
    # set necrosis to 1
    np_seg[1 == np_seg] = 1.
    # set proliferative tumor to 1
    np_seg[4 == np_seg] = 1.
    seg_field = load_field(DG0, np_seg, 'init', pixel_scale)

    # mean diffusivity
    md_path = create_path(directory, patient_id, timepoint, data='md')
    np_md = load_nii(md_path, offsets_left, offsets_right)
    md_field = load_field(DG0, np_md, 'md', pixel_scale)

    return mesh, wm_field, gm_field, csf_field, seg_field, md_field


def assemble_system_ch(V, tau, mobility, eps, landa, C_psi, phi_now):
    W = [V, V]
    phi, mu = map(df.TrialFunction, W)
    psi, nu = map(df.TestFunction, W)

    tau, eps2, C_psi = map(df.Constant, [tau, eps**2, C_psi])

    a = [[0, 0], [0, 0]]
    a[0][0] = + eps2*df.inner(df.grad(phi), df.grad(psi))*df.dx + 3 * C_psi * phi*psi*df.dx
    a[0][1] = - mu * psi * df.dx
    a[1][0] = - phi * nu * df.dx
    a[1][1] = - tau * df.inner(mobility * df.grad(mu), df.grad(nu))*df.dx

    l = [0, 0]
    l[0] = - C_psi * (4 * phi_now**3 - 6 * phi_now**2 - phi_now) * psi * df.dx
    l[1] = - phi_now*nu*df.dx

    #l[1] += - tau * 21 * phi_now * (1 - phi_now) * nu * df.dx
    #l[1] += - df.Constant(landa) * tau/eps * phi_now * (1 - phi_now) * nu * df.dx
    #l[1] += - tau/eps * phi_now * (1 - phi_now) * nu * df.dx

    A, b = map(block_assemble, [a, l])

    return A, b


def assemble_preconditioner_ch(V, tau, mobility, eps, C_psi, phi_now):
    W = [V, V]
    phi, mu = map(df.TrialFunction, W)
    psi, nu = map(df.TestFunction, W)

    sqtau = math.sqrt(tau)

    tau, eps2, C_psi, sqtau = map(df.Constant, [tau, eps**2, C_psi, sqtau])

    a11 = 0 
    a11 += tau * df.inner(mobility * df.grad(mu), df.grad(nu)) * df.dx
    a11 += sqtau * mu * nu * df.dx
    A11 = df.assemble(a11)

    a00 = 0
    a00 += eps2 * df.inner(df.grad(phi), df.grad(psi)) * df.dx
    a00 += 3 * C_psi / sqtau * psi * phi * df.dx
    A00 = df.assemble(a00)

    A = block_mat([
        [LU(A00), 0],
        [0, LU(A11)],
    ])

    return A


class ReactionEquation(df.NonlinearProblem):
    def __init__(self, a, L):
        df.NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        df.assemble(self.L, tensor=b)
    def J(self, A, x):
        df.assemble(self.a, tensor=A)


class RandomInitialConditions(df.UserExpression):
    """ Random initial conditions """
    def __init__(self, mean, delta, **kwargs):
        self.mean = mean
        self.delta = delta 
        random.seed(2 + df.MPI.rank(df.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.mean + self.delta/2*(0.5 - random.random())


def ExpressionSmoothCircle(midpoint: df.Point, r: float, a: float, s: float, degree: int):
    r_str = f'std::sqrt(std::pow(x[0]-({midpoint[0]}),2) + std::pow(x[1]-({midpoint[1]}),2))'
    return df.Expression(f's / (exp( a * ({r_str}-r) ) + 1)', r=r, a=a, s=s, degree=degree)


def ufl_positive(x):
    #value is set to 0 if negative and to 1 if larger than 1
    y = df.conditional(df.ge(x,1.0),df.Constant(1.0),x) #if x is larger than 1, set to 1
    z = df.conditional(df.ge(y,0.0),y,df.Constant(0.0)) #if y is less than 0, then set to 0
    return z


def solve_reaction(V, phi_vec, tau, eps, landa):
    phi = df.Function(V)
    phi.vector()[:] = phi_vec[:]
    phi_old = phi.copy(deepcopy=True)
    phi_old.vector()[:] = phi_vec[:]
    psi = df.TestFunction(V)
    dx = df.dx
    F = (df.inner(phi, psi) - df.inner(phi_old, psi) - 0.5 * landa * df.Constant(tau/eps) * df.inner(phi * (1 - phi), psi) - 0.5 * landa * df.Constant(tau/eps) * df.inner(phi_old * (1 - phi_old), psi)) * dx
    df.solve(F == 0, phi, [], solver_parameters={'newton_solver': {'relative_tolerance': 1e-12}})
    phi_vec[:] = phi.vector()[:]
    return phi


def _demo():
    parser = argparse.ArgumentParser(prog='Cahn Hilliard simulator for exporting mobilities to test preconditioners')
    parser.add_argument('--Nout', type=int, help='Multiple of time step width for which we write the output.', default=1)
    parser.add_argument('--tau', help='Time step width.', default=1)
    parser.add_argument('--eps', help='Interfacial width.', default=4e-3)
    parser.add_argument('--t-end', help='When does the simulation stop?', default=100)
    parser.add_argument('--export-mobility', action='store_true')

    args = parser.parse_args()

    mesh, wm_field, gm_field, csf_field, seg_field, md_field = load_2d_slice()

    df.File('output/wm_field.pvd') << wm_field
    df.File('output/gm_field.pvd') << gm_field
    df.File('output/seg_field.pvd') << seg_field
    df.File('output/md_field.pvd') << md_field

    mobility_scale = 1.

    phi_mobility = lambda phi: 8 * ufl_positive(phi)**2 * (1-ufl_positive(phi))**2
    matter_mobility = md_field * mobility_scale
    mobility = lambda phi: matter_mobility * phi_mobility(phi)

    V = df.FunctionSpace(mesh, 'P', 1)
    phi_now = df.interpolate(seg_field, V)
    phi_now.rename('phi_now', '')

    tau = args.tau 
    eps = args.eps 
    C_psi = 1. 
    t_end = args.t_end 
    landa = 5e-4

    file_phi = df.File('output/phi_now.pvd')

    t = 0   # current time in simulation
    it = 0  # iteration number

    landa_fun = landa
    
    def write_output(t):
        ''' Utility function to save the simulation state.'''
        file_phi << phi_now, t
    
    u_next = block_vec([phi_now.vector(), 0]) 

    # save initial state
    write_output(t)

    phi_mob = df.Function(V)

    while t < t_end:
        solve_reaction(V, phi_now.vector(), tau/2, eps, landa_fun)

        phi_mob.vector()[:] = phi_now.vector()[:]
        A, b = assemble_system_ch(V, tau, mobility(phi_now), eps, landa_fun, C_psi, phi_now)
        Pinv = assemble_preconditioner_ch(V, tau, mobility(phi_now), eps, C_psi, phi_now)
        Ainv = MinRes(A, precond=Pinv, initial_guess=u_next, show=1)
        u_next = Ainv * b

        phi_now.vector()[:] = u_next.blocks[0]

        solve_reaction(V, phi_now.vector(), tau/2, eps, landa_fun)

        mass1 = df.assemble(df.inner(np.abs(phi_now), 1) * df.dx)
        mass2 = df.assemble(df.inner(phi_now, 1) * df.dx)
        print(f'{t}: {mass1} {mass2}')

        t += tau
        it += 1

        if it % args.Nout == 0:
            write_output(t)


if __name__ == '__main__':
    _demo()