import math
import sys
from copy import copy
from operator import itemgetter

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mp_baselines.planners.base import MPPlanner
from mp_baselines.planners.rrt_base import RRTBase
from mp_baselines.planners.utils import safe_path, purge_duplicates_from_traj, extend_path
from torch_robotics.torch_utils.torch_timer import TimerCUDA


class TreeNode:

    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def render(self, ax):
        assert ax is not None, "Axis cannot be None"
        if self.parent is not None:
            if ax.name == '3d':
                x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
                ax.plot3D([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color='k', linewidth=0.5)
            else:
                x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
                ax.plot([x[0], y[0]], [x[1], y[1]], color='k', linewidth=0.5)

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'
    __repr__ = __str__


def random_swap(nodes1, nodes2):
    p = float(len(nodes1)) / (len(nodes1) + len(nodes2))
    swap = (torch.rand(1) < p)
    return swap


def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))


class RRTConnect(RRTBase):

    def __init__(
            self,
            task=None,
            n_iters: int = None,
            start_state_pos: torch.Tensor = None,
            step_size: float = 0.1,
            n_radius: float = 1.,
            max_time: float = 60.,
            goal_state_pos: torch.Tensor = None,
            tensor_args: dict = None,
            n_pre_samples=10000,
            pre_samples=None,
            **kwargs
    ):
        super(RRTConnect, self).__init__(
            'RRTConnect',
            task,
            n_iters,
            start_state_pos,
            goal_state_pos,
            step_size,
            n_radius,
            max_time,
            tensor_args,
            n_pre_samples,
            pre_samples
        )

        self.nodes_tree_1 = None
        self.nodes_tree_2 = None

        self.nodes_tree_1_torch = None
        self.nodes_tree_2_torch = None

    def _run_optimization(self, opt_iters, **observation):
        """
        Run optimization iterations.
        """
        print_freq = observation.get('print_freq', 150)
        debug = observation.get('debug', False)

        if self.collision_fn(self.start_state_pos).squeeze() or self.collision_fn(self.goal_state_pos).squeeze():
            return None

        iteration = -1
        success = False

        self.nodes_tree_1 = [TreeNode(self.start_state_pos)]
        self.nodes_tree_2 = [TreeNode(self.goal_state_pos)]

        self.nodes_tree_1_torch = self.nodes_tree_1[0].config
        self.nodes_tree_2_torch = self.nodes_tree_2[0].config

        path = None

        with TimerCUDA() as t:
            while (t.elapsed < self.max_time) and (iteration < self.n_iters):
                iteration += 1

                if iteration % print_freq == 0 or iteration % (self.n_iters - 1) == 0:
                    if debug:
                        self.print_info(iteration, t.elapsed, success)

                # Swap trees
                # TODO - implement smart random swap
                # swap = random_swap(self.nodes_tree_1, self.nodes_tree_2)
                swap = True
                if swap:
                    self.nodes_tree_1, self.nodes_tree_2 = self.nodes_tree_2, self.nodes_tree_1
                    self.nodes_tree_1_torch, self.nodes_tree_2_torch = self.nodes_tree_2_torch, self.nodes_tree_1_torch

                # Sample new node
                target = self.sample_fn(**observation)

                ###############################################################
                # nearest node in Tree1 to the target node
                nearest = self.get_nearest_node(self.nodes_tree_1, self.nodes_tree_1_torch, target)

                # create a safe path from the target node to the nearest node
                extended = self.extend_fn(nearest.config, target, max_step=self.step_size, max_dist=self.n_radius)
                path = safe_path(extended, self.collision_fn)

                # add last node in path to Tree1
                if len(path) == 0:
                    continue

                n1 = TreeNode(path[-1], parent=nearest)
                self.nodes_tree_1.append(n1)
                self.nodes_tree_1_torch = torch.vstack((self.nodes_tree_1_torch, n1.config))

                if torch.allclose(path[-1], target):
                    self.remove_last_pre_sample()

                ###############################################################
                # add nearest node in Tree2 to the one recently added to Tree1
                nearest = self.get_nearest_node(self.nodes_tree_2, self.nodes_tree_2_torch, n1.config)

                # create a safe path from the target node to the nearest node
                extended = self.extend_fn(nearest.config, n1.config, max_step=self.step_size, max_dist=self.n_radius)
                path = safe_path(extended, self.collision_fn)

                # add last node in path to TREE2
                if len(path) == 0:
                    continue

                n2 = TreeNode(path[-1], parent=nearest)
                self.nodes_tree_2.append(n2)
                self.nodes_tree_2_torch = torch.vstack((self.nodes_tree_2_torch, n2.config))

                if swap:
                    self.nodes_tree_1, self.nodes_tree_2 = self.nodes_tree_2, self.nodes_tree_1
                    self.nodes_tree_1_torch, self.nodes_tree_2_torch = self.nodes_tree_2_torch, self.nodes_tree_1_torch

                # if the last node in path is the same as the proposed node, the two trees are connected and terminate
                if torch.allclose(n1.config, n2.config):
                    success = True

                    if swap:
                        n1, n2 = n2, n1

                    path1, path2 = n1.retrace(), n2.retrace()

                    # if swap:
                    #     path1, path2 = path2, path1

                    path = configs(path1[:-1] + path2[::-1])
                    break

        if path is not None:
            if len(path) == 1:
                return None
            self.print_info(iteration, t.elapsed, success)
            return purge_duplicates_from_traj(path, eps=1e-6)
        return path

    def print_info(self, iteration, elapsed_time, success):
        print(f'Iteration: {iteration:5}/{self.n_iters:5} '
              f'| Time: {elapsed_time:.3f} s'
              f'| Nodes: {len(self.nodes_tree_1) + len(self.nodes_tree_2)} '
              f'| Success: {success}'
              )

    def render(self, ax, **kwargs):
        for node in self.nodes_tree_1:
            node.render(ax)
        for node in self.nodes_tree_2:
            node.render(ax)
