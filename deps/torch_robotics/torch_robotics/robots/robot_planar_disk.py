"""
MIT License

Copyright (c) 2024 Yorai Shaoul. Heavily based on the code robot_point_mass.py from MPD.

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
from typing import Tuple

# Standard imports.
import numpy as np
import torch
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt

from mmd.common.constraints import *
from mmd.common.trajectory_utils import *
# Project imports.
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy, tensor_linspace_v1, to_torch
from mmd.config import MMDParams as params


class RobotPlanarDisk(RobotBase):

    # def __init__(self,
    #              name='RobotPlanarDisk',
    #              radius=0.07,
    #              q_limits=torch.tensor([[-1, -1], [1, 1]]),  # Confspace limits.
    #              **kwargs):
    #     super().__init__(
    #         name=name,
    #         q_limits=to_torch(q_limits, **kwargs['tensor_args']),
    #         **kwargs
    #     )
    #
    #     ################################################################################################
    #     # Robot
    #     self.radius = radius
    #     # Set the link margins for object collision checking to the radius of the disk here and in the parent.
    #     self.link_margins_for_object_collision_checking = [radius]

    def __init__(self,
                 name='RobotPlanarDisk',
                 radius=params.robot_planar_disk_radius,
                 q_limits=torch.tensor([[-1, -1], [1, 1]]),  # configuration space limits
                 **kwargs):
        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            link_names_for_object_collision_checking=['link_0'],
            link_margins_for_object_collision_checking=[radius * 1.1],
            link_idxs_for_object_collision_checking=[0],
            num_interpolated_points_for_object_collision_checking=1,
            **kwargs
        )
        self.radius = radius

    def render(self, ax, q=None, color='blue', cmap='Blues', margin_multiplier=1., **kwargs):
        if q is not None:
            margin = self.radius * margin_multiplier * 0.9  # The 0.9 margin multiplier is to make the disks display correctly.
            q = to_numpy(q)
            if q.ndim == 1:
                if self.q_dim == 2:
                    circle1 = plt.Circle(q, margin, color=color, zorder=10)
                    ax.add_patch(circle1)
                    if 'q_tail' in kwargs:
                        n_tail = len(kwargs['q_tail'])
                        q_tail = kwargs['q_tail']
                        for i, q2 in enumerate(kwargs['q_tail']):
                            margin_tail = (n_tail - i) / n_tail * margin
                            alpha = (1 - i / n_tail) / 2
                            circle2 = plt.Circle(q2, margin_tail, color=color, zorder=10, alpha=alpha, edgecolor=None)
                            ax.add_patch(circle2)
                elif self.q_dim == 3:
                    plot_sphere(ax, q, np.zeros_like(q), margin, cmap)
                else:
                    raise NotImplementedError
            elif q.ndim == 2:
                if q.shape[-1] == 2:
                    # ax.scatter(q[:, 0], q[:, 1], color=color, s=10 ** 2, zorder=10)
                    circ = []
                    for q_ in q:
                        circ.append(plt.Circle(q_, margin, color=color))
                        coll = mcoll.PatchCollection(circ, zorder=10)
                        ax.add_collection(coll)
                elif q.shape[-1] == 3:
                    # ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=color, s=10 ** 2, zorder=10)
                    for q_ in q:
                        plot_sphere(ax, q_, np.zeros_like(q_), margin, cmap)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', constraints_l=None, **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            trajs_np = to_numpy(trajs_pos)
            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 3))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 2))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color]*segment.shape[0])
                # ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)
        if start_state is not None:
            start_state_np = to_numpy(start_state)
            if len(start_state_np) == 3:
                ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=self.radius*100)
            else:
                ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=self.radius*100)
        if goal_state is not None:
            goal_state_np = to_numpy(goal_state)
            if len(goal_state_np) == 3:
                ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], marker='o',
                        color='purple', markersize=self.radius*100)
            else:
                ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=self.radius*100)
        if constraints_l is not None:
            for c in constraints_l:
                # if isinstance(c, MultiPointConstraint):
                q_l = c.get_q_l()
                q_l_np = [to_numpy(q) for q in q_l]
                for q in q_l_np:
                    if self.q_dim == 3:
                        plot_sphere(ax, q, np.zeros_like(q), self.radius, 'Blues')
                    else:
                        # circle = plt.Circle(q, self.radius, color='red', zorder=10)
                        # ax.add_patch(circle)
                        # Show gaussian centered at this point.
                        for i in range(1, 50):
                            circle = plt.Circle(q, self.radius * (i / 50)**4, color='red', zorder=10, alpha=0.2 * (50 - i)/50)
                            ax.add_patch(circle)

                # elif isinstance(c, VertexConstraint):
                #     q = c.get_q()
                # ...
    def fk_map_collision_impl(self, q, **kwargs):
        # There is no forward kinematics. Assume it's the identity.
        # Add tasks space dimension
        return q.unsqueeze(-2)

    def check_rr_collisions(self, robot_q: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Check collisions between robots. (Robot-robot collisions).
        Args:
            robot_q: (..., n_robots, q_dim)
        Returns:
            collisions: (..., n_robots, n_robots), True if there is a collision between the robots.
            collision_points: (..., n_robots, n_robots, 2 or 3), the collision points.
        """
        # # Check collisions between robots.
        # return are_points_closer_than_margin(robot_q, 2.1 * self.radius)

        points_batch = robot_q
        margin = 2.1 * self.radius  # ///////////////
        assert points_batch.dim() >= 2
        points_batch1 = points_batch.unsqueeze(-2)
        points_batch2 = points_batch.unsqueeze(-3)
        # Pairwise distances between robots.
        robot_dq = points_batch1 - points_batch2
        # (..., n_robots, n_robots)
        robot_dq_norm = torch.norm(robot_dq, dim=-1)
        # (..., n_robots, n_robots)
        collisions = robot_dq_norm < margin
        # Set the trace to be False.
        collisions = collisions & ~torch.eye(collisions.shape[-1], device=collisions.device, dtype=collisions.dtype)
        # Collision points. For this robot those are the midpoint between the robots.
        collision_points = (points_batch1 + points_batch2) / 2
        # Mask the collision points. Set the collision points to nan if there is no collision.
        collision_points = collision_points * collisions.unsqueeze(-1)
        collision_points[~collisions.unsqueeze(-1).expand_as(collision_points)] = float('nan')
        return collisions, collision_points
