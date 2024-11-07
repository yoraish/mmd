# MIT License

# Copyright (c) 2022 An Thai Le, João Carvalho

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# **********************************************************************
# The first version of some files were licensed as
# "Original Source License" (see below). Several enhancements and bug fixes
# were done by An Thai Le, João Carvalho since obtaining the first version.



# Original Source License:

# MIT License

# Copyright (c) 2022 Sasha Lambert

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch


class OccupancyMap:
    """
    Generates an occupancy grid.
    """
    def __init__(self, map_dim, cell_size, tensor_args=None):

        for d in map_dim:
            assert d % 2 == 0, f"Dimension {d} is not an even number"

        self.map_dim = map_dim

        self.n_dim = len(map_dim)

        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.cmap_dim = [0 for _ in map_dim]
        for i, d in enumerate(self.cmap_dim):
            self.cmap_dim[i] = ceil(map_dim[i]/cell_size)

        self.map = np.zeros(self.cmap_dim)
        self.map_torch = None

        self.cell_size = cell_size

        # Map center (in cells)
        self.origin = np.array([d//2 for d in self.cmap_dim])
        self.c_offset = torch.Tensor(self.origin).to(**tensor_args)

        # limits
        self.dims = self.map.shape
        self.lims = [(-self.cell_size * d/2, self.cell_size * d/2) for d in self.dims]

    def convert_map_to_torch(self):
        self.map_torch = torch.Tensor(self.map).to(**self.tensor_args)
        return self.map_torch

    def get_collisions(self, X, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, horizon, task_spaces, position_dim)
        :return: collision cost on the trajectories
        """

        X_occ = X * (1/self.cell_size) + self.c_offset
        X_occ = X_occ.floor()

        # X_occ = X_occ.cpu().numpy().astype(np.int)
        X_occ = X_occ.type(torch.LongTensor)
        X_occ = X_occ.to(device=self.tensor_args['device'])

        # Project out-of-bounds locations to axis
        for i in range(X_occ.shape[-1]):
            X_occ[..., i] = X_occ[..., i].clamp(0, self.map.shape[i]-1)

        # Collisions
        try:
            if X_occ.shape[-1] == 2:
                collision_vals = self.map_torch[X_occ[..., 1], X_occ[..., 0]]
            elif X_occ.shape[-1] == 3:
                collision_vals = self.map_torch[X_occ[..., 1], X_occ[..., 0], X_occ[..., 2]]
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            print(X_occ)
            print(X_occ.clamp(0, self.map.shape[0]-1))
        return collision_vals

    def compute_distances(self, X, **kwargs):
        """
        Computes euclidean distances of X to all points in the occupied grid
        """
        X_grid_points_idxs = torch.nonzero(self.map_torch > 0)
        X_grid_points_task_space = (X_grid_points_idxs - self.c_offset) * self.cell_size
        distances = torch.cdist(X, X_grid_points_task_space, p=2.0)
        return distances

    def compute_cost(self, X, **kwargs):
        return self.get_collisions(X, **kwargs)

    def zero_grad(self):
        pass

    def plot(self, ax=None, save_dir=None, filename="obst_map.png"):
        if ax is None:
            if self.n_dim == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

        if self.n_dim == 2:
            res = self.map.shape[0]
            x = np.linspace(self.lims[0][0], self.lims[0][1], res)
            y = np.linspace(self.lims[1][0], self.lims[1][1], res)
            ax.contourf(x, y, np.clip(self.map, 0, 1), 2, cmap='Greys')
        else:
            x, y, z = np.indices(np.array(self.map.shape) + 1, dtype=float)
            x -= self.origin[0]
            x = x * self.cell_size
            y -= self.origin[1]
            y = y * self.cell_size
            z -= self.origin[2]
            z = z * self.cell_size
            ax.voxels(y, x, z, self.map, facecolors='gold', edgecolor='black', shade=False, alpha=0.05)

