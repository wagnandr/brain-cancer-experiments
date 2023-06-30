'''
Small collection of routines to read in Bene's tumor data.
As packages we use 
- nibabel to read in nii.gz, so call nifti files, and
- dipy to calculate the diffusion tensor.
'''
import os
import numpy as np
import nibabel as nib


def create_path(directory='data-bwohlmuth', patient_id='008', timepoint='preop', data='dti', type='nii.gz'):
    """
    Small utility script to generate paths, which fit Bene's naming scheme.
    """
    mainpath = os.path.join(directory, f'respond_tum_{patient_id}', f'{timepoint}')
    filename = f'sub-respond_tum_{patient_id}_ses-{timepoint}_space-sri_{data}.{type}'
    return os.path.join(mainpath, filename)


def calculate_diffusion_tensor(directory='data-bwohlmuth', patient_id='008', timepoint='preop'):
    """
    Calculates the diffusion tensor from diffusion tensor imaging information and saves it on disk.
    This takes a while, hence should not be called lightly. :P 
    In dipy we basically solve
        S_i / S_0 = exp(- b g^T D g)
    for D, where the observations S_i, S_0 are saved in the nii.gz files, and b and g are in the bval and bvec files.
    """
    # since dipy is just needed here (and will at some point not be needed anymore), wo only import it locally:
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    # load the directions g and scaling factors b from disk to create a gradient table
    path_bval, path_bvec = [create_path(directory, patient_id, timepoint, data='dti', type=t) for t in ('bval', 'bvec')]
    bvals, bvecs = read_bvals_bvecs(path_bval, path_bvec)
    gtable = gradient_table(bvals, bvecs)
    # load the dti info from the disk
    path_dti = create_path(directory, patient_id, timepoint, data='dti', type='nii.gz')
    img_dti = nib.load(path_dti)
    img_dti_data = img_dti.get_fdata()
    # load the brainmask from the disk
    path_bmask = create_path(directory, patient_id, timepoint, data='brainmask', type='nii.gz')
    img_bmask_data = nib.load(path_bmask).get_fdata()
    # solve for the diffusion tensor
    tenmodel = dti.TensorModel(gtable)
    tenfit = tenmodel.fit(img_dti_data, img_bmask_data)
    # save to disk
    lower_triangular_nifty = nib.Nifti1Image(tenfit.lower_triangular(), img_dti.affine)
    path_tensor = create_path(directory, patient_id, timepoint, data='tensorlt', type='nii.gz') 
    nib.save(lower_triangular_nifty, path_tensor)


def load_diffusion_tensor(directory='data-bwohlmuth', patient_id='008', timepoint='preop'):
    """
    Loads a diffusion tensor which was calculated by calculate diffusion tensor.
    """
    path_tensor = create_path(directory, patient_id, timepoint, data='tensorlt', type='nii.gz') 
    tensor = nib.load(path_tensor).get_fdata()
    # For ordering see https://github.com/dipy/dipy/blob/321e06722ef42b5add3a7f570f6422845177eafa/dipy/reconst/dti.py?#L2030-L2031
    Dxx, Dxy, Dyy, Dxz, Dyz, Dzz = [tensor[:,:,:,i] for i in range(6)]
    D = [[0]*3 for i in range(3)]
    D[0][0] = Dxx
    D[1][1] = Dyy
    D[2][2] = Dzz
    D[0][1] = D[1][0] = Dxy
    D[0][2] = D[2][0] = Dxz
    D[1][2] = D[2][1] = Dyz
    return D


def load_nii(path, offsets_left=(0,0,0), offsets_right=(None, None, None)):
    """Loads only a small slice from a nifty file to save save space."""
    nii_field = nib.load(path)
    np_field = nii_field.get_fdata()
    offl, offr = offsets_left, offsets_right
    np_field = np.squeeze(np_field[offl[0]:offr[0], offl[1]:offr[1], offl[2]:offr[2]])
    return np_field


def get_bounds(field):
    """Returns the bounds, in which our nifity field is different from zero"""
    Nx, Ny, Nz = field.shape
    x = np.arange(0,Nx)
    y = np.arange(0,Ny)
    z = np.arange(0,Nz)
    xx, yy, zz = np.meshgrid(x,y,z)
    xx_masked = xx[field > 0]
    xbounds = xx_masked.min(), xx_masked.max()
    yy_masked = yy[field > 0]
    ybounds = yy_masked.min(), yy_masked.max()
    zz_masked = zz[field > 0]
    zbounds = zz_masked.min(), zz_masked.max()
    return np.array([xbounds, ybounds, zbounds])


def _demo_diffusion_tensor():
    """Small demo to generate and read in the diffusion tensor to test if everything is alright."""
    import argparse
    parser = argparse.ArgumentParser(description='Creates diffusion tensors.')
    parser.add_argument('--directory', help='Base path of the directory structure provided by Bene.', default='data-bwohlmuth')
    parser.add_argument('--patient-id', help='The patient id.', default='008')
    parser.add_argument('--timepoint', help='The point in time when the examination is done.', choices=('preop', 'postop', 'recurrence'), default='preop')
    parser.add_argument('--create-all', help='Creates all the diffusion tensors of all the patients, if requested.', action='store_true')
    args = parser.parse_args()
    if args.create_all:
        for patient_id in ['008', '019', '020']:
            for timepoint in ('preop', 'postop', 'recurrence'): 
                print(f'Started patient {patient_id} at {timepoint}.')
                try:
                    calculate_diffusion_tensor(args.directory, patient_id, timepoint)
                except:
                    print(f'Failed patient {patient_id} at {timepoint}.')
                print(f'Succeeded patient {patient_id} at {timepoint}.')
    else: 
        calculate_diffusion_tensor(args.directory, args.patient_id, args.timepoint)
        D = load_diffusion_tensor(args.directory, args.patient_id, args.timepoint)
        print(f'D[0][0] = {D[0][0]}')


if __name__ == '__main__':
    _demo_diffusion_tensor()
