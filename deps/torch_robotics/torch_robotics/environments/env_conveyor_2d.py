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
# General imports.
import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import List
# Project imports.
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvConveyor2D(EnvBase):

    def __init__(self,
                 name='EnvConveyor2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):

        obj_list = [
            MultiSphereField(
                np.array([]),  # (n, 2) array of sphere centers.
                np.array([]),  # (n, ) array of sphere radii.
                tensor_args=tensor_args
            ),
            MultiBoxField(
                np.array([
                    [0, 0],
                    [0, 0.35],
                    [0, -0.35]
                ]),
                np.array([
                    [0.8, 0.1],
                    [1.0, 0.1],
                    [1.0, 0.1]
                ]),
                tensor_args=tensor_args
            ),

        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # Environments limits.
            obj_fixed_list=[ObjectField(obj_list, 'conveyor2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.05,
            n_pre_samples=50000,
            max_time=50
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=2,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params
        else:
            raise NotImplementedError

    def get_skill_pos_seq_l(self, robot=None, start_pos=None, goal_pos=None) -> List[torch.Tensor]:
        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            # Options for position sequences that all paths must go through (each path must choose one). This neeed not be uniformly discretized. The optimizer
            # and interpolator will take care of that.
            skill0_start = torch.tensor([-0.6, -0.2], **self.tensor_args)
            skill0_end = torch.tensor([0.6, -0.2], **self.tensor_args)
            skill0 = torch.stack(
                [torch.lerp(skill0_start, skill0_end, alpha) for alpha in torch.linspace(0, 1, 30, **self.tensor_args)])
            skill1_start = torch.tensor([0.6, 0.2], **self.tensor_args)
            skill1_end = torch.tensor([-0.6, 0.2], **self.tensor_args)
            skill1 = torch.stack(
                [torch.lerp(skill1_start, skill1_end, alpha) for alpha in torch.linspace(0, 1, 30, **self.tensor_args)])
            skill_pos_sequence_l = [skill0, skill1]
            return skill_pos_sequence_l
        else:
            raise NotImplementedError

    def compute_traj_data_adherence(self, path: torch.Tensor):
        # Compute the data adherence of the path.
        # A trajectory is said to be "adhering" if it has either
        # 1. Entered the top corridor from the right and left from the left.
        # 2. Entered the bottom corridor from the left and left from the right.
        # Iterate through the path and find the time, if any, agent passes
        # through entrance, middle, and exit of a corridor.
        t_top_corridor = torch.tensor([-3, -2, -1], **self.tensor_args)  # t_enter, t_middle, t_exit
        t_bottom_corridor = torch.tensor([-3, -2, -1], **self.tensor_args)  # t_enter, t_middle, t_exit
        q_top_corridor = torch.tensor([[0.6, 0.2], [0.0, 0.2], [-0.6, 0.2]], **self.tensor_args)
        q_bottom_corridor = torch.tensor([[-0.6, -0.2], [0.0, -0.2], [0.6, -0.2]], **self.tensor_args)
        for t in range(path.shape[0]):
            # Always only check the next available waypoint.
            next_available_ix_top = torch.argmin(t_top_corridor)
            q_top_next = q_top_corridor[next_available_ix_top, :]
            if torch.norm(path[t, :] - q_top_next) < 0.2:
                t_top_corridor[next_available_ix_top] = t
            next_available_ix_bottom = torch.argmin(t_bottom_corridor)
            q_bottom_next = q_bottom_corridor[next_available_ix_bottom, :]
            if torch.norm(path[t, :] - q_bottom_next) < 0.2:
                t_bottom_corridor[next_available_ix_bottom] = t
        if torch.all(t_top_corridor != -1) or torch.all(t_bottom_corridor != -1):
            return 1
        else:
            return 0


if __name__ == '__main__':
    env = EnvConveyor2D(
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
