import numpy as np

from torch_robotics.environments.primitives import MultiSphereField, ObjectField, MultiBoxField


def create_grid_spheres(rows=5, cols=5, heights=0, radius=0.1, distance_from_border=0.1, tensor_args=None):
    # Generates a grid (rows, cols, heights) of circles
    # if heights = 0, creates a 2d grid, else a 3d grid
    dim = 2 if heights == 0 else 3
    centers_x = np.linspace(-1 + distance_from_border, 1 - distance_from_border, cols)
    centers_y = np.linspace(-1 + distance_from_border, 1 - distance_from_border, rows)
    z_flat = None
    if dim == 3:
        centers_z = np.linspace(-1 + distance_from_border, 1 - distance_from_border, heights)
        X, Y, Z = np.meshgrid(centers_x, centers_y, centers_z)
        z_flat = Z.flatten()
    else:
        X, Y = np.meshgrid(centers_x, centers_y)

    flats = [X.flatten(), Y.flatten()]
    if z_flat:
        flats.append(z_flat)
    centers = np.array(flats).T
    radii = np.ones(flats[0].shape[0]) * radius

    spheres = MultiSphereField(centers, radii, tensor_args=tensor_args)
    obj_field = ObjectField([spheres], 'grid-of-spheres')
    obj_list = [obj_field]
    return obj_list
