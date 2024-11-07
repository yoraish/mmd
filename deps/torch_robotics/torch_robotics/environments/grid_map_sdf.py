from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.functional import jacobian


class GridMapSDF:
    """
    Generates an SDF grid.
    """
    def __init__(self, limits, cell_size, obj_list, tensor_args=None):

        self.limits = limits
        self.dim = limits.shape[-1]

        self.tensor_args = tensor_args

        # objects
        self.obj_list = obj_list

        # SDF grids
        map_dim = torch.abs(limits[1] - limits[0])
        self.map_dim = map_dim
        self.cell_size = cell_size
        self.cmap_dim = torch.ceil(map_dim/cell_size).type(torch.LongTensor).to(self.tensor_args['device'])

        self.points_for_sdf = None
        self.sdf_tensor = None
        self.grad_sdf_tensor = None
        self.precompute_sdf()

    def precompute_sdf(self):
        # create voxel grid of points
        basis_ranges = [
            torch.linspace(self.limits[0][0], self.limits[1][0], self.cmap_dim[0], **self.tensor_args),
            torch.linspace(self.limits[0][1], self.limits[1][1], self.cmap_dim[1], **self.tensor_args),
        ]
        if self.dim == 3:
            basis_ranges.append(
                torch.linspace(self.limits[0][2], self.limits[1][2], self.cmap_dim[2], **self.tensor_args)
            )
        points_for_sdf_meshgrid = torch.meshgrid(*basis_ranges, indexing='ij')
        self.points_for_sdf = torch.stack(points_for_sdf_meshgrid, dim=-1)

        f_grad_sdf = lambda x: self.compute_signed_distance_raw(x).sum()
        sdf_tensor_l = []
        grad_sdf_tensor_l = []
        # TODO - compute sdf and gradient in batches to prevent memory overflow
        for i in range(self.points_for_sdf.shape[0]):
            torch.cuda.empty_cache()
            # sdf
            points_sdf = self.points_for_sdf[i]
            sdf_tensor = self.compute_signed_distance_raw(points_sdf)
            sdf_tensor_l.append(sdf_tensor)
            # gradient of sdf
            grad_sdf_tensor = jacobian(f_grad_sdf, points_sdf)
            grad_sdf_tensor_l.append(grad_sdf_tensor)
        torch.cuda.empty_cache()

        self.sdf_tensor = torch.stack(sdf_tensor_l, dim=0)
        self.grad_sdf_tensor = torch.stack(grad_sdf_tensor_l, dim=0)

    def compute_signed_distance_raw(self, x):
        sdf = None
        for obj in self.obj_list:
            sdf_obj = obj.compute_signed_distance(x)
            if sdf is None:
                sdf = sdf_obj
            else:
                sdf = torch.minimum(sdf, sdf_obj)
        return sdf

    def __call__(self, X, **kwargs):
        return self.get_sdf(X, **kwargs)

    def compute_cost(self, X, **kwargs):
        return self.get_sdf(X, **kwargs)

    def compute_signed_distance(self, X, **kwargs):
        return self.get_sdf(X, **kwargs)

    def get_sdf(self, X, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, horizon, task_spaces, position_dim)
        :return: collision cost on the trajectories
        """
        X_in_map = ((X-self.limits[0])/self.map_dim * self.cmap_dim).floor().to(torch.int)

        # Project out-of-bounds locations to axis
        max_idx = torch.tensor(self.points_for_sdf.shape[:-1]).type_as(X_in_map) - 1
        X_in_map = X_in_map.clamp(torch.zeros_like(max_idx), max_idx)

        # SDFs and gradients
        # To compute the gradients, because we already have computed the gradients, we use a surrogate sdf function
        # surrogate_sdf(x) = sdf(x_detachted) + x @ grad_sdf(x_detachted) - x_detached @ grad_sdf(x_detachted)
        # This surrogate has the property
        # surrogate_sdf(x) = sdf(x_detachted)
        # grad_x_surrogate_sdf(x) = grad_sdf(x_detachted)
        X_in_map_detached = X_in_map.detach()
        X_query = torch.unbind(X_in_map_detached, dim=-1)

        sdf_vals = self.sdf_tensor[X_query]
        grad_sdf = self.grad_sdf_tensor[X_query]

        X_detached = X.detach()
        sdf_vals += (X * grad_sdf).sum(-1) - (X_detached * grad_sdf).sum(-1)

        return sdf_vals

    def zero_grad(self):
        pass
