"""
MIT License

Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard includes.
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian

# Project includes.
from torch_robotics.environments import EnvHighways2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvHighways2DExtraObjects(EnvHighways2D):

    def __init__(self, tensor_args=None, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                # np.array([[0.5, 0.5]]),  # (n, 2) array of sphere centers.
                np.array([]),  # (n, 2) array of sphere centers.
                # np.array([.04]),  # (n, ) array of sphere radii.
                np.array([]),  # (n, ) array of sphere radii.
                tensor_args=tensor_args
            ),
            # MultiBoxField(
            #     np.array(  # (n, 2) array of box centers.
            #         [
            #             [0.0, -0.0],
            #         ]
            #     ),
            #     np.array(  # (n, 2) array of box sizes.
            #         [
            #             [0.4, 0.2],
            #         ]
            #     ),
            #     tensor_args=tensor_args
            # )
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'conveyor2d-extraobjects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvHighways2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
