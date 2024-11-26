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
from collections import OrderedDict

# Standard imports.
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.collections as mcoll

# Project imports.
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch


class RobotCompositeThreePlanarDisk(RobotBase):
    def __init__(self,
                 name='RobotCompositeThreePlanarDisk',
                 radius=0.05,
                 q_limits=torch.tensor([
                     [-1, -1, -1, -1, -1, -1],
                     [ 1,  1,  1,  1,  1,  1]
                 ]),  # configuration space limits
                 **kwargs):
        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            link_names_for_object_collision_checking=['link_0', 'link_1', 'link_2'],
            link_margins_for_object_collision_checking=[radius * 1.1] * 3,  # Inflate so that self-collision affects robots slightly further away.
            link_idxs_for_object_collision_checking=[0, 1, 2],
            num_interpolated_points_for_object_collision_checking=3,
            link_idxs_for_self_collision_checking=[0, 1, 2],
            self_collision_margin_robot=radius * 4,  # Inflated.
            link_names_for_self_collision_checking=['link_0', 'link_1', 'link_2'],
            num_interpolated_points_for_self_collision_checking=3,
            link_names_pairs_for_self_collision_checking=OrderedDict({
                'link_0': ['link_1', 'link_2'],
                'link_1': ['link_0', 'link_2'],
                'link_2': ['link_0', 'link_1'],
            }),
            **kwargs
        )
        self.radius = radius
        self.n_agents = 3

    def render(self, ax, q=None, color='blue', cmap='Blues', margin_multiplier=1., **kwargs):
        q_multi = q
        if q_multi is not None:
            margin = self.radius * margin_multiplier * 0.9  # The 0.9 margin multiplier is to make the disks display correctly.
            q_multi = to_numpy(q_multi)
            if q_multi.ndim == 1:
                if self.q_dim == self.n_agents * 2:
                    for agent_id in range(self.n_agents):
                        q = q_multi[agent_id * 2: (agent_id + 1) * 2]
                        circle1 = plt.Circle(q, margin, color=color, zorder=10)
                        ax.add_patch(circle1)
                        if 'q_tail' in kwargs:
                            n_tail = len(kwargs['q_tail'])
                            for i, q2 in enumerate(kwargs['q_tail']):
                                margin_tail = (n_tail - i) / n_tail * margin
                                alpha = (1 - i / n_tail) / 2
                                for agent_id2 in range(self.n_agents):
                                    if agent_id2 != agent_id:
                                        q2_single = q2[agent_id2 * 2: (agent_id2 + 1) * 2]
                                        circle2 = plt.Circle(q2_single, margin_tail, color=color, zorder=10, alpha=alpha, edgecolor=None)
                                        ax.add_patch(circle2)

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(
            self, ax, trajs=None, start_state=None, goal_state=None, colors=['blue'],
            linestyle='solid', **kwargs):
        if trajs is not None:
            trajs_pos_multi = self.get_position(trajs)
            trajs_np_multi = to_numpy(trajs_pos_multi)
            if self.q_dim == 6:
                for agent_id in range(self.n_agents):
                    trajs_np = trajs_np_multi[:, :, agent_id * 2: (agent_id + 1) * 2]
                    segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                    line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle)
                    ax.add_collection(line_segments)
                    points = np.reshape(trajs_np, (-1, 2))
                    colors_scatter = []
                    for segment, color in zip(segments, colors):
                        colors_scatter.extend([color]*segment.shape[0])
                    # ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)
        if start_state is not None:
            for i in range(self.n_agents):
                start_state_np = to_numpy(start_state[i*2:(i+1)*2])
                ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=self.radius*100)
        if goal_state is not None:
            for i in range(self.n_agents):
                goal_state_np = to_numpy(goal_state[i*2:(i+1)*2])
                ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=self.radius*100)

    def fk_map_collision_impl(self, q, **kwargs):
        # There is no forward kinematics. Return the centers of the disks in a (B, H, n_robots, 2) tensor.
        B, H, q_dim = q.shape
        q = q.view(B, H, self.n_agents, 2)
        return q
