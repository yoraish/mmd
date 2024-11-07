from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil
from typing import List

import einops
import numpy as np
import torch
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import FancyBboxPatch, BoxStyle
from torch.autograd.functional import jacobian

from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_rotation_matrix
from torch_robotics.torch_kinematics_tree.geometrics.utils import transform_point, rotate_point
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, to_numpy, tensor_linspace_v1
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes



class PrimitiveShapeField(ABC):
    """
    Represents a primitive object in N-D.
    """

    def __init__(self, dim=3, tensor_args=None):
        self.dim = dim
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS
        self.tensor_args = tensor_args

    def compute_signed_distance(self, x):
        """
        Returns the signed distance at x.
        Parameters
        ----------
            x : torch array (batch, dim) or (batch, horizon, dim)
        """
        return self.compute_signed_distance_impl(x)

    @abstractmethod
    def compute_signed_distance_impl(self, x):
        # The SDF computed here assumes the primitive shape main center is located at the origin, and there is no
        # rotation.
        # Note that this center is different from e.g. the centers for spheres in MultiSphereField, since a primitive
        # shape can be made of multiple shapes.
        raise NotImplementedError()

    def zero_grad(self):
        pass

    def obstacle_collision_check(self, obst_map):
        valid = True
        obst_map_test = self.add_to_occupancy_map(deepcopy(obst_map))
        if np.any(obst_map_test.map > 1):
            valid = False
        return valid

    def point_collision_check(self, obst_map, pts):
        valid = True
        if pts is not None:
            obst_map_test = self.add_to_occupancy_map(np.copy(obst_map))
            for pt in pts:
                if obst_map_test[ceil(pt[0]), ceil(pt[1])] >= 1:
                    valid = False
                    break
        return valid

    @abstractmethod
    def add_to_occupancy_map(self, obst_map):
        # Adds obstacle to an occupancy grid
        raise NotImplementedError

    @abstractmethod
    def render(self, ax, pos=None, ori=None, color=None, **kwargs):
        raise NotImplementedError


def plot_sphere(ax, center, pos, radius, cmap):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = radius * (np.cos(u) * np.sin(v))
    y = radius * (np.sin(u) * np.sin(v))
    z = radius * np.cos(v)
    ax.plot_surface(
        x + center[0] + pos[0], y + center[1] + pos[1], z + center[2] + pos[2],
        cmap=cmap,
        alpha=1
    )


class MultiSphereField(PrimitiveShapeField):

    def __init__(self, centers, radii, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Centers of the spheres.
            radii : numpy array
                Radii of the spheres.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.radii = to_torch(radii, **self.tensor_args)

    def __repr__(self):
        return f"MultiSphereField(centers={self.centers}, radii={self.radii})"

    def compute_signed_distance_impl(self, x):
        if len(self.centers) == 0:
            return torch.ones_like(x[..., 0])

        distance_to_centers = torch.norm(x.unsqueeze(-2) - self.centers.unsqueeze(0), dim=-1)
        # sdfs = distance_to_centers - self.radii.unsqueeze(0)
        sdfs = distance_to_centers - self.radii
        return torch.min(sdfs, dim=-1)[0]

    def zero_grad(self):
        self.centers.grad = None
        self.radii.grad = None

    def add_to_occupancy_map(self, obst_map):
        # Adds obstacle to occupancy map
        for center, radius in zip(self.centers, self.radii):
            n_dim = len(center)
            # Convert dims to cell indices
            c_r = ceil(radius / obst_map.cell_size)
            c_x = ceil(center[0] / obst_map.cell_size)
            c_y = ceil(center[1] / obst_map.cell_size)

            # centers in cell indices
            obst_map_origin_xi = obst_map.origin[0]
            obst_map_origin_yi = obst_map.origin[1]

            c_x_cell = c_x + obst_map_origin_xi
            c_y_cell = c_y + obst_map_origin_yi

            obst_map_x_dim = obst_map.dims[0]
            obst_map_y_dim = obst_map.dims[1]

            if n_dim == 3:
                c_z = ceil(center[2] / obst_map.cell_size)
                obst_map_origin_zi = obst_map.origin[2]
                c_z_cell = c_z + obst_map_origin_zi
                obst_map_z_dim = obst_map.dims[2]

            for i in range(c_x_cell - 2 * c_r, c_x_cell + 2 * c_r):
                if i < 0 or i >= obst_map_x_dim:
                    continue
                for j in range(c_y_cell - 2 * c_r, c_y_cell + 2 * c_r):
                    if j < 0 or j >= obst_map_y_dim:
                        continue

                    if n_dim == 3:
                        for k in range(c_z_cell - 2 * c_r, c_z_cell + 2 * c_r):
                            if k < 0 or k >= obst_map_z_dim:
                                continue

                            p = torch.tensor([(i - obst_map_origin_xi) * obst_map.cell_size,
                                              (j - obst_map_origin_yi) * obst_map.cell_size,
                                              (k - obst_map_origin_zi) * obst_map.cell_size
                                              ],
                                             **self.tensor_args)

                            if self.is_inside(p, center, radius):
                                obst_map.map[j, i, k] += 1

                    else:
                        p = torch.tensor([(i - obst_map_origin_xi) * obst_map.cell_size,
                                          (j - obst_map_origin_yi) * obst_map.cell_size],
                                         **self.tensor_args)

                        if self.is_inside(p, center, radius):
                            obst_map.map[j, i] += 1
        return obst_map

    @staticmethod
    def is_inside(p, center, radius):
        # Check if point p is inside the discretized sphere
        return torch.linalg.norm(p - center) <= radius

    def render(self, ax, pos=None, ori=None, color='wheat', cmap='wheat', **kwargs):
        for center, radius in zip(self.centers, self.radii):
            center = to_numpy(center)
            radius = to_numpy(radius)
            if pos is None:
                pos = np.zeros(self.dim)
            else:
                pos = to_numpy(pos)
            # orientation is not needed, because the shape is symmetric
            if ax.name == '3d':
                plot_sphere(ax, center, pos, radius, cmap)
            else:
                circle = plt.Circle((center[0] + pos[0], center[1] + pos[1]), radius, color=color, linewidth=0, alpha=1)
                ax.add_patch(circle)


def patch_rotate_translate(ax, patch, rot, trans):
    rot_angle_deg = np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0]))
    affine_transf = transforms.Affine2D().rotate_deg(rot_angle_deg).translate(trans[0], trans[1]) + ax.transData
    patch.set_transform(affine_transf)
    ax.add_patch(patch)


class MultiBoxField(PrimitiveShapeField):

    def __init__(self, centers, sizes, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Center of the boxes.
            sizes: numpy array
                Sizes of the boxes.
        """
        super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
        self.centers = to_torch(centers, **self.tensor_args)
        self.sizes = to_torch(sizes, **self.tensor_args)
        self.half_sizes = self.sizes / 2

    def __repr__(self):
        return f"MultiBoxField(centers={self.centers}, sizes={self.sizes})"

    def compute_signed_distance_impl(self, x):
        distance_to_centers = torch.abs(x.unsqueeze(-2) - self.centers.unsqueeze(0))
        sdfs = torch.max(distance_to_centers - self.half_sizes.unsqueeze(0), dim=-1)[0]
        return torch.min(sdfs, dim=-1)[0]

    def zero_grad(self):
        self.centers.grad = None
        self.sizes.grad = None
        self.half_sizes.grad = None

    def add_to_occupancy_map(self, obst_map):
        for center, size in zip(self.centers, self.sizes):
            n_dim = len(center)
            width = size[0]
            height = size[1]
            # Convert dims to cell indices
            w = ceil(width / obst_map.cell_size)
            h = ceil(height / obst_map.cell_size)
            c_x = ceil(center[0] / obst_map.cell_size)
            c_y = ceil(center[1] / obst_map.cell_size)

            obst_map_origin_xi = obst_map.origin[0]
            obst_map_origin_yi = obst_map.origin[1]

            if n_dim == 2:
                obst_map.map[
                c_y - ceil(h / 2.) + obst_map_origin_yi:c_y + ceil(h / 2.) + obst_map_origin_yi,
                c_x - ceil(w / 2.) + obst_map_origin_xi:c_x + ceil(w / 2.) + obst_map_origin_xi
                ] += 1
            else:
                depth = size[2]
                c_z = ceil(center[2] / obst_map.cell_size)
                obst_map_origin_zi = obst_map.origin[2]
                d = ceil(depth / obst_map.cell_size)
                obst_map.map[
                c_y - ceil(h / 2.) + obst_map_origin_yi:c_y + ceil(h / 2.) + obst_map_origin_yi,
                c_x - ceil(w / 2.) + obst_map_origin_xi:c_x + ceil(w / 2.) + obst_map_origin_xi,
                c_z - ceil(d / 2.) + obst_map_origin_zi:c_z + ceil(d / 2.) + obst_map_origin_zi
                ] += 1
        return obst_map

    def get_cube(self):
        phi = torch.arange(1, 10, 2, **self.tensor_args) * torch.pi / 4
        Phi, Theta = torch.meshgrid(phi, phi, indexing="ij")
        x = torch.cos(Phi) * torch.sin(Theta)
        y = torch.sin(Phi) * torch.sin(Theta)
        z = torch.cos(Theta) / np.sqrt(2)
        return x, y, z

    def render(self, ax, pos=None, ori=None, color='wheat', cmap='gray', **kwargs):

        rot = q_to_rotation_matrix(ori).squeeze()
        if ax.name == '3d':
            x, y, z = self.get_cube()
            for center, size in zip(self.centers, self.sizes):
                cx, cy, cz = center
                a, b, c = size

                points_x = cx + x * a
                points_y = cy + y * b
                points_z = cz + z * c

                points = torch.stack((points_x.ravel(), points_y.ravel(), points_z.ravel()), dim=-1)
                points = transform_point(points, rot, pos)

                d = x.shape[0]
                points_x = points[:, 0].view(d, d)
                points_y = points[:, 1].view(d, d)
                points_z = points[:, 2].view(d, d)

                points_x_np, points_y_np, points_z_np = to_numpy(points_x), to_numpy(points_y), to_numpy(points_z)
                # TODO - implemented drawing of rounded boxes in 3D
                ax.plot_surface(points_x_np, points_y_np, points_z_np, cmap=cmap, alpha=0.25)
        else:
            for i, (center, size) in enumerate(zip(self.centers, self.sizes)):
                cx, cy = to_numpy(center)
                a, b = to_numpy(size)
                pos_np = to_numpy(pos)
                # by definition a rotation in the xy-plane is around the z-axis
                rot_np = to_numpy(rot[:2, :2])
                point = np.array([cx - a / 2, cy - b / 2])
                self.draw_box(ax, i, point, a, b, rot_np, pos_np[:2], color)

    def draw_box(self, ax, i, point, a, b, rot, trans, color='wheat'):
        rectangle = plt.Rectangle((point[0], point[1]), a, b,
                                  color=color, linewidth=0, alpha=1)
        patch_rotate_translate(ax, rectangle, rot, trans)


class MultiRoundedBoxField(MultiBoxField):

    def __init__(self, centers, sizes, tensor_args=None):
        """
        Parameters
        ----------
            centers : numpy array
                Center of the boxes.
            sizes: numpy array
                Sizes of the boxes.
        """
        super().__init__(centers, sizes, tensor_args=tensor_args)
        self.radius = torch.min(self.sizes, dim=-1)[0] * 0.15  # empirical value

    def compute_signed_distance_impl(self, x):
        # Implementation of rounded box
        # https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html
        distance_to_centers = torch.abs(x.unsqueeze(-2) - self.centers.unsqueeze(0))
        q = distance_to_centers - self.half_sizes.unsqueeze(0) + self.radius.unsqueeze(0).unsqueeze(-1)
        max_q = torch.amax(q, dim=-1)
        sdfs = torch.minimum(max_q, torch.zeros_like(max_q)) + torch.linalg.norm(torch.relu(q), dim=-1) - self.radius.unsqueeze(0)
        return torch.min(sdfs, dim=-1)[0]

    def draw_box(self, ax, i, point, a, b, rot, trans, color='wheat'):
        rounded_box = FancyBboxPatch((point[0], point[1]), a, b, color=color,
                                     boxstyle=BoxStyle.Round(pad=0., rounding_size=to_numpy(self.radius[i]).item())
                                     )
        patch_rotate_translate(ax, rounded_box, rot, trans)


# Alias for rounded box.
# Use a rounded box instead of a box by default.
# This creates smoother cost functions, which are important to gradient-based optimization methods.
MultiBoxField = MultiRoundedBoxField


# TODO - NEEDS CHECKING
# class MultiInfiniteCylinderField(PrimitiveShapeField):
#
#     def __init__(self, centers, radii, tensor_args=None):
#         """
#         Parameters
#         ----------
#             centers : numpy array
#                 Centers of the cylinders.
#             radii : numpy array
#                 Radii of the cylinders.
#         """
#         super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
#         self.centers = torch.tensor(centers, **self.tensor_args)
#         self.radii = torch.tensor(radii, **self.tensor_args)
#
#     def __repr__(self):
#         return f"InfiniteCylinder(centers={self.centers}, radii={self.radii})"
#
#     def compute_signed_distance_impl(self, x):
#         # treat it like a circle in 2d
#         distance_to_centers_2d = torch.norm(x[..., :2].unsqueeze(-2) - self.centers.unsqueeze(0)[..., :2], dim=-1)
#         sdfs = distance_to_centers_2d - self.radii.unsqueeze(0)
#         return torch.min(sdfs, dim=-1)[0]
#
#     def zero_grad(self):
#         self.centers.grad = None
#         self.radii.grad = None
#
#     def add_to_occupancy_map(self, obst_map):
#         raise NotImplementedError
#
#     def render(self, ax, pos=None, ori=None):
#         raise NotImplementedError
#         # https://stackoverflow.com/a/49311446
#         def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
#             z = np.linspace(0, height_z, 50)
#             theta = np.linspace(0, 2 * np.pi, 50)
#             theta_grid, z_grid = np.meshgrid(theta, z)
#             x_grid = radius * np.cos(theta_grid) + center_x
#             y_grid = radius * np.sin(theta_grid) + center_y
#             return x_grid, y_grid, z_grid
#
#         for center, radius in zip(self.centers, self.radii):
#             cx, cy, cz = to_numpy(center)
#             radius = to_numpy(radius)
#             xc, yc, zc = data_for_cylinder_along_z(cx, cy, radius, 1.5)  # add height just for visualization
#             ax.plot_surface(xc, yc, zc, cmap='wheat', alpha=0.75)
#
#
# class MultiCylinderField(MultiInfiniteCylinderField):
#
#     def __init__(self, centers, radii, heights, tensor_args=None):
#         """
#         Parameters
#         ----------
#             centers : numpy array
#                 Centers of the cylinders.
#             radii : numpy array
#                 Radii of the cylinders.
#             heights : numpy array
#                 Heights of the cylinders.
#         """
#         super().__init__(centers, radii, tensor_args=tensor_args)
#         self.heights = torch.tensor(heights, **self.tensor_args)
#         self.half_heights = self.heights / 2
#
#     def __repr__(self):
#         return f"Cylinder(centers={self.centers}, radii={self.radii}, heights={self.heights})"
#
#     def compute_signed_distance_impl(self, x):
#         raise NotImplementedError
#         x = x - self.center
#         x_proj = x[:, :2]
#         x_proj_norm = torch.norm(x_proj, dim=-1)
#         x_proj_norm = torch.where(x_proj_norm > 0, x_proj_norm, torch.ones_like(x_proj_norm))
#         x_proj = x_proj / x_proj_norm[:, None]
#         x_proj = x_proj * self.radius
#         x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
#         return torch.norm(x - x_proj, dim=-1) - self.radius
#
#     def zero_grad(self):
#         super().zero_grad()
#         self.heights.grad = None
#
#     def add_to_occupancy_map(self, obst_map):
#         raise NotImplementedError
#
#
# class MultiEllipsoidField(PrimitiveShapeField):
#
#     def __init__(self, centers, radii, tensor_args=None):
#         """
#         Axis aligned ellipsoid.
#         Parameters
#         ----------
#             center : numpy array
#                 Center of the ellipsoid.
#             radii : numpy array
#                 Radii of the ellipsoid.
#         """
#         super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
#         self.center = torch.tensor(centers, **self.tensor_args)
#         self.radii = torch.tensor(radii, **self.tensor_args)
#
#     def compute_signed_distance_impl(self, x):
#         return torch.norm((x - self.center) / self.radii, dim=-1) - 1
#
#     def zero_grad(self):
#         self.center.grad = None
#         self.radii.grad = None
#
#     def __repr__(self):
#         return f"Ellipsoid(center={self.center}, radii={self.radii})"
#
#
# class MultiCapsuleField(PrimitiveShapeField):
#
#     def __init__(self, centers, radii, heights, tensor_args=None):
#         """
#         Parameters
#         ----------
#             centers : numpy array
#                 Center of the capsule.
#             radiii : float
#                 Radius of the capsule.
#             heights : float
#                 Height of the capsule.
#         """
#         super().__init__(dim=centers.shape[-1], tensor_args=tensor_args)
#         self.center = torch.tensor(centers, **self.tensor_args)
#         self.radius = torch.tensor(radii, **self.tensor_args)
#         self.height = torch.tensor(heights, **self.tensor_args)
#
#     def compute_signed_distance_impl(self, x):
#         x = x - self.center
#         x_proj = x[:, :2]
#         x_proj_norm = torch.norm(x_proj, dim=-1)
#         x_proj_norm = torch.where(x_proj_norm > 0, x_proj_norm, torch.ones_like(x_proj_norm))
#         x_proj = x_proj / x_proj_norm[:, None]
#         x_proj = x_proj * self.radius
#         x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
#         x_proj = torch.norm(x - x_proj, dim=-1) - self.radius
#         x_proj = torch.where(x_proj > 0, x_proj, torch.zeros_like(x_proj))
#         x_proj = torch.where(x_proj < self.height, x_proj, torch.ones_like(x_proj) * self.height)
#         x_proj = torch.cat([x_proj, x[:, 2:]], dim=-1)
#         return torch.norm(x - x_proj, dim=-1) - self.radius
#
#     def zero_grad(self):
#         self.center.grad = None
#         self.radius.grad = None
#         self.height.grad = None
#
#     def __repr__(self):
#         return f"Capsule(center={self.center}, radius={self.radius}, height={self.height})"
#
#
# class MeshField(PrimitiveShapeField):
#     """
#     Represents a mesh as a primitive shape.
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)


########################################################################################################################
class ObjectField(PrimitiveShapeField):

    def __init__(self, primitive_fields, name='object', pos=None, ori=None, reference_frame='base'):
        """
        Holds an object made of primitives and manages its position and orientation in the environments.
        """
        self.name = name

        assert primitive_fields is not None
        assert isinstance(primitive_fields, List)
        super().__init__(dim=primitive_fields[0].dim, tensor_args=primitive_fields[0].tensor_args)
        self.fields = primitive_fields

        # position and orientation
        assert (pos is None and ori is None) or (pos.nelement() == 3 and ori.nelement() == 4)
        self.pos = torch.zeros(3, **self.tensor_args) if pos is None else pos
        self.ori = torch.tensor([1, 0, 0, 0], **self.tensor_args) if ori is None else ori  # quat - wxyz

        # Reference frame for the position and orientation
        self.reference_frame = reference_frame

        # precomputed sdf and gradients
        # holds the sdf (and its gradient) of this object for all points in the environments workspace
        self.grid_map_sdf = None

    def __repr__(self):
        return f"ObjectField(fields={self.fields})"

    def set_position_orientation(self, pos=None, ori=None):
        if pos is not None:
            assert len(pos) == 3
            self.pos = to_torch(pos, **self.tensor_args)
        if ori is not None:
            assert len(ori) == 4, "quaternion wxyz"
            self.ori = to_torch(ori, **self.tensor_args)

    def join_primitives(self):
        raise NotImplementedError

    def compute_signed_distance_impl(self, x):
        # Transform the point before computing the SDF.
        # The implemented SDFs assume the objects are centered around 0 and not rotated.
        x_shape = x.shape
        if x_shape[-1] == 2:
            x_new = torch.cat((x, torch.zeros((x.shape[:-1]), **self.tensor_args).unsqueeze(-1)), dim=-1)
        else:
            x_new = x

        # transform back to the origin
        rot = q_to_rotation_matrix(self.ori).squeeze().transpose(-2, -1)
        x_new = rotate_point(x_new-self.pos, rot)
        if x_shape[-1] == 2:
            x_new = x_new[..., :2]

        sdf_fields = []
        for field in self.fields:
            sdf_fields.append(field.compute_signed_distance_impl(x_new))
        return torch.min(torch.stack(sdf_fields, dim=-1), dim=-1)[0]

    def render(self, ax, pos=None, ori=None, color='wheat', **kwargs):
        for field in self.fields:
            if pos is None:
                pos = self.pos
            if ori is None:
                ori = self.ori
            field.render(ax, pos=pos, ori=ori, color=color, **kwargs)

    def add_to_occupancy_map(self, occ_map):
        for field in self.fields:
            field.add_to_occupancy_map(occ_map)

    def zero_grad(self):
        for field in self.fields:
            field.zero_grad()


if __name__ == '__main__':
    tensor_args = DEFAULT_TENSOR_ARGS
    spheres = MultiSphereField(torch.zeros(2, **tensor_args).view(1, -1),
                               torch.ones(1, **tensor_args).view(1, -1) * 0.3,
                               tensor_args=tensor_args)

    boxes = MultiBoxField(torch.zeros(2, **tensor_args).view(1, -1) + 0.5,
                          torch.ones(2, **tensor_args).view(1, -1) * 0.3,
                          tensor_args=tensor_args)

    obj_field = ObjectField([spheres, boxes])

    theta = np.deg2rad(45)
    # obj_field.set_position_orientation(pos=[-0.5, 0., 0.])
    # obj_field.set_position_orientation(ori=[np.cos(theta/2), 0, 0, np.sin(theta/2)])
    obj_field.set_position_orientation(pos=[-0.5, 0., 0.], ori=[np.cos(theta/2), 0, 0, np.sin(theta/2)])

    # # Render objects
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    obj_field.render(ax)
    plt.show()

    # Render sdf
    fig, ax = plt.subplots()
    xs = torch.linspace(-1, 1, steps=400, **tensor_args)
    ys = torch.linspace(-1, 1, steps=400, **tensor_args)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    X_flat = torch.flatten(X)
    Y_flat = torch.flatten(Y)
    sdf = obj_field.compute_signed_distance(torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, 2))
    sdf = sdf.reshape(X.shape)
    sdf_np = to_numpy(sdf)
    ctf = ax.contourf(to_numpy(X), to_numpy(Y), sdf_np)
    fig.colorbar(ctf, orientation='vertical')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Render gradient sdf
    xs = torch.linspace(-1, 1, steps=20, **tensor_args)
    ys = torch.linspace(-1, 1, steps=20, **tensor_args)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    X_flat = torch.flatten(X)
    Y_flat = torch.flatten(Y)
    stacked_tensors = torch.stack((X_flat, Y_flat), dim=-1).view(-1, 1, 2)

    f_grad_sdf = lambda x: obj_field.compute_signed_distance(x).sum()
    grad_sdf = jacobian(f_grad_sdf, stacked_tensors)

    grad_sdf_np = to_numpy(grad_sdf).squeeze()
    ax.quiver(to_numpy(X_flat), to_numpy(Y_flat),
              grad_sdf_np[:, 0], grad_sdf_np[:, 1],
              color='red')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()
