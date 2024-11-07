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
from mmd.common.trajectory_utils import densify_trajs


class EnvHighways2D(EnvBase):

    def __init__(self,
                 name='EnvHighways2D',
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
                    [0, 0.0],
                    [0., 0.875],
                    [0., -0.875],
                    [0.875, 0.0],
                    [-0.875, 0.0],
                    [0.875, 0.875],
                    [0.875, -0.875],
                    [-0.875, 0.875],
                    [-0.875, -0.875],
                ]),
                np.array([
                    [0.5, 0.5],
                    [0.5, 0.25],
                    [0.5, 0.25],
                    [0.25, 0.5],
                    [0.25, 0.5],
                    [0.25, 0.25],
                    [0.25, 0.25],
                    [0.25, 0.25],
                    [0.25, 0.25]
                ]),
                tensor_args=tensor_args
            ),
        ]
        # np.array([
        #     [0, 0.0],
        #     [0., 0.875],
        #     [0., -0.875],
        #     [0.875, 0.0],
        #     [-0.875, 0.0]
        # ]),
        # np.array([
        #     [0.7, 0.7],
        #     [0.7, 0.35],
        #     [0.7, 0.35],
        #     [0.35, 0.7],
        #     [0.35, 0.7]
        # ]),
        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # Environments limits.
            obj_fixed_list=[ObjectField(obj_list, 'highways2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=500,
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
            opt_iters=20,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=1e-5,
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

    def is_start_goal_valid_for_data_gen(self, robot, start_pos, goal_pos):
        """
        :param robot: Robot object.
        :param start_pos: Start position. (q_dim,), often (2, ).
        :param goal_pos: Goal position. (q_dim,), often (2, ).
        """
        if torch.linalg.norm(start_pos - goal_pos) > 0.6:
            return False
        # We set squares for starts and goals. Starts can only be at start squares and goals follow similarly.
        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            start_region_centers = torch.tensor([
                [0.8, 0.5],
                [-0.5, 0.8],
                [-0.8, -0.5],
                [0.5, -0.8]
            ], **self.tensor_args)
            goal_region_centers = torch.tensor([
                [0.8, -0.5],
                [0.5, 0.8],
                [-0.8, 0.5],
                [-0.5, -0.8]
            ], **self.tensor_args)
            start_region_radius = 0.15
            goal_region_radius = 0.15
            if torch.any(torch.norm(start_region_centers - start_pos, dim=-1) < start_region_radius).item() and \
               torch.any(torch.norm(goal_region_centers - goal_pos, dim=-1) < goal_region_radius).item():
                return True
            else:
                return False

    def get_skill_pos_seq_l(self, robot=None, start_pos=None, goal_pos=None) -> List[torch.Tensor]:
        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):

            # These are quadrant-midpoints.
            ordered_waypoints = torch.tensor([
                [-0.5, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5]
            ], **self.tensor_args)

            # Find the closest waypoint to the start.
            start_pos = start_pos.unsqueeze(0)
            ordered_waypoints = ordered_waypoints.unsqueeze(1)
            distances = torch.norm(ordered_waypoints - start_pos, dim=-1)
            closest_waypoint_start_idx = torch.argmin(distances, dim=0).item()
            closest_waypoint_start = ordered_waypoints[closest_waypoint_start_idx].squeeze(0)

            # Find the closest waypoint to the goal.
            goal_pos = goal_pos.unsqueeze(0)
            distances = torch.norm(ordered_waypoints - goal_pos, dim=-1)
            closest_waypoint_goal_idx = torch.argmin(distances, dim=0).item()
            closest_waypoint_goal = ordered_waypoints[closest_waypoint_goal_idx].squeeze(0)

            # Create a skill that goes from closest entrance to closest exit via the other waypoints order.
            skill_pos_sequence = [closest_waypoint_start]
            if closest_waypoint_start_idx == closest_waypoint_goal_idx:
                closest_waypoint_start_idx = (closest_waypoint_start_idx + 1) % len(ordered_waypoints)
            while closest_waypoint_start_idx != closest_waypoint_goal_idx:
                closest_waypoint_start_idx = (closest_waypoint_start_idx + 1) % len(ordered_waypoints)
                skill_pos_sequence.append(ordered_waypoints[closest_waypoint_start_idx].squeeze(0))
            # skill_pos_sequence.append(closest_waypoint_goal)
            skill_pos_sequence_l = [torch.stack(skill_pos_sequence)]

            skill_pos_sequence_l = densify_trajs(skill_pos_sequence_l, n_points_interp=10)
            # Get rid of the first few points and the last few points. We only care about the homotopy.
            skill_pos_sequence_l = [skill_pos_sequence_l[0][4:-4]]
            # Add noise.
            skill_pos_sequence_l += [skill_pos_sequence_l[i] + torch.randn_like(skill_pos_sequence_l[i]) * 0.01 for i in range(len(skill_pos_sequence_l))]

            # Show the skill.
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # print(skill_pos_sequence_l[0])
            # ax.plot(skill_pos_sequence_l[0][:, 0].detach().cpu().detach().cpu().numpy(), skill_pos_sequence_l[0][:, 1].detach().cpu().numpy(), 'r')
            # ax.scatter(start_pos[0, 0].item(), start_pos[0, 1].item(), c='g')
            # ax.scatter(goal_pos[0, 0].item(), goal_pos[0, 1].item(), c='b')
            # ax.set_xlim(-1, 1)
            # ax.set_ylim(-1, 1)
            # plt.show()

            return skill_pos_sequence_l
        else:
            raise NotImplementedError

    def compute_traj_data_adherence(self, path: torch.Tensor):
        # Compute vectors between consecutive points
        vectors = path[:,:]

        # Normalize the vectors
        norms = torch.norm(vectors, dim=1, keepdim=True)
        vectors_normalized = vectors / norms

        # Get pairs of consecutive normalized vectors
        vec1 = vectors_normalized[:-1]
        vec2 = vectors_normalized[1:]

        # Compute the 2D cross products of consecutive vectors
        cross_products = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]

        # Sum the cross products and determine if counterclockwise or clockwise
        aggregate_cross_product = torch.sum(cross_products)

        return 1 if aggregate_cross_product > 0 else 0

if __name__ == '__main__':
    env = EnvHighways2D(
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
