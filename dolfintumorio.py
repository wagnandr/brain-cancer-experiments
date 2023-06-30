import dolfin as df


class MatterField(df.UserExpression):
    """Represents a numpy grid as a UserExpression"""
    def __init__(self, np_field, pixel_scale, **kwargs):
        super().__init__(**kwargs)
        self.np_field = np_field
        self.pixel_scale = pixel_scale

    def eval(self, values, x):
        if len(self.np_field.shape) == 2:
            values[0] = self.np_field[int(x[0] / self.pixel_scale), int(x[1] / self.pixel_scale)] 
        elif len(self.np_field.shape) == 3:
            values[0] = self.np_field[int(x[0] / self.pixel_scale), int(x[1] / self.pixel_scale), int(x[2] / self.pixel_scale)] 
        else:
            raise RuntimeError('Cannot understand the structure of the numpy field.')


def load_field(
        V: df.FunctionSpace, 
        numpy_data,
        name: str,
        pixel_scale: float):
    """
    Loads a single field from a numpy field via the MatterField class and renames it accordingly 
    """
    field = df.interpolate(MatterField(numpy_data, pixel_scale), V)
    field.rename(name, name)
    return field