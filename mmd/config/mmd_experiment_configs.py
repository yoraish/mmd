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
from abc import ABC, abstractmethod
import numpy as np
import torch

# Project imports.
from mmd.config.mmd_params import MMDParams as params
from mmd.common.multi_agent_utils import *
from torch_robotics.environments import *
from torch_robotics.environments.env_highways_2d import EnvHighways2D


def get_planning_problem(planning_problem_class_name: str,
                         num_agents: int):
    # Get the planning problem.
    planning_problem_class = globals()[planning_problem_class_name]
    planning_problem = planning_problem_class()
    return planning_problem.get_planning_problem(num_agents)


class MMDPlanningProblemConfig(ABC):
    # Some parameters.
    name = ""

    @abstractmethod
    def get_planning_problem(self, num_agents):
        pass


class EnvEmpty2DRobotPlanarDiskCircle(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvEmpty2D_RobotPlanarDisk_Circle"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_circle(num_agents, radius=0.8)
        global_model_ids = [['EnvEmpty2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvEmpty2DRobotPlanarDiskRandom(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvEmpty2D_RobotPlanarDisk_Random"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(num_agents=num_agents,
                                                                               env_class=EnvEmpty2D,
                                                                               tensor_args=params.tensor_args,
                                                                               margin=0.15)
        global_model_ids = [['EnvEmpty2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvHighways2DRobotPlanarDiskRandom(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvHighways2D_RobotPlanarDisk_Random"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(num_agents=num_agents,
                                                                               env_class=EnvHighways2D,
                                                                               tensor_args=params.tensor_args,
                                                                               margin=0.15)
        global_model_ids = [['EnvHighways2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvEmpty2DRobotPlanarDiskBoundary(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvEmpty2D_RobotPlanarDisk_Circle"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_boundary(num_agents, dist=0.87)
        global_model_ids = [['EnvEmpty2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvConveyor2DRobotPlanarDiskBoundary(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvConveyor2D_RobotPlanarDisk_Boundary"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_boundary(num_agents, dist=0.87)
        global_model_ids = [['EnvConveyor2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvConveyor2DRobotPlanarDiskRandom(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvConveyor2D_RobotPlanarDisk_Random"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(num_agents=num_agents,
                                                                                 env_class=EnvConveyor2D,
                                                                                 tensor_args=params.tensor_args,
                                                                                 margin=0.15)
        global_model_ids = [['EnvConveyor2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvDropRegion2DRobotPlanarDiskRandom(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvDropRegion2D_RobotPlanarDisk_Random"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(num_agents=num_agents,
                                                                                 env_class=EnvDropRegion2D,
                                                                                 tensor_args=params.tensor_args,
                                                                                 margin=0.15)
        global_model_ids = [['EnvDropRegion2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvHighways2DRobotPlanarDiskSmallCircle(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvHighways2D_RobotPlanarDisk_SmallCircle"

    def get_planning_problem(self, num_agents):

        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_circle(min(num_agents, 10), radius=0.45)
        if num_agents > 10:
            more_start_state_pos_l, more_goal_state_pos_l = get_start_goal_pos_circle(num_agents - 10, radius=0.65)
            start_state_pos_l += more_start_state_pos_l
            goal_state_pos_l += more_goal_state_pos_l

        global_model_ids = [['EnvHighways2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvDropRegion2DRobotPlanarDiskBoundary(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvDropRegion2D_RobotPlanarDisk_Boundary"

    def get_planning_problem(self, num_agents):
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_boundary(num_agents)
        global_model_ids = [['EnvDropRegion2D-RobotPlanarDisk']]
        agent_skeleton_l = [[[0, 0]]] * num_agents
        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvTestTwoByTwoRobotPlanarDiskRandom(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvTestTwoByTwo_RobotPlanarDisk_Random"
        self.start_state_pos_l, self.goal_state_pos_l = get_start_goal_pos_circle(10, radius=0.45)
        more_start_state_pos_l, more_goal_state_pos_l = get_start_goal_pos_circle(20, radius=0.65)
        self.start_state_pos_l += more_start_state_pos_l
        self.goal_state_pos_l += more_goal_state_pos_l

        self.global_model_ids = [['EnvEmptyNoWait2D-RobotPlanarDisk', 'EnvConveyor2D-RobotPlanarDisk'],
                                 ['EnvHighways2D-RobotPlanarDisk', 'EnvHighways2D-RobotPlanarDisk']]
        self.agent_skeleton_options = [[[0, 0], [0, 1], [1, 1]],
                                       [[0, 0], [1, 0], [1, 1]],
                                       [[1, 0], [0, 0], [1, 0]],
                                       [[0, 0], [0, 1], [1, 1]],
                                       [[0, 0], [0, 1], [0, 0]],
                                       [[1, 1], [0, 1], [0, 0]],
                                       [[1, 1], [0, 1], [0, 0]],
                                       [[1, 0], [1, 1], [1, 0]],
                                       [[1, 1], [1, 0], [0, 0]],
                                       [[0, 0], [1, 0], [0, 0]],
                                       [[1, 0], [0, 0], [1, 0]],
                                       [[1, 1], [0, 1], [1, 1]],
                                       [[1, 1], [1, 0], [1, 1]],
                                       [[0, 0], [1, 0], [1, 1]],
                                       [[1, 0], [1, 1], [1, 0]],
                                       [[0, 0], [0, 1], [1, 1]],
                                       [[1, 0], [0, 0], [0, 1]],
                                       [[1, 0], [1, 1], [1, 0]],
                                       [[1, 1], [1, 0], [0, 0]],
                                       [[1, 1], [0, 1], [1, 1]],
                                       [[1, 1], [1, 0], [1, 1]],
                                       [[1, 0], [1, 1], [0, 1]],
                                       [[1, 0], [0, 0], [1, 0]],
                                       [[1, 1], [1, 0], [0, 0]],
                                       [[1, 1], [0, 1], [0, 0]],
                                       [[0, 0], [1, 0], [1, 1]],
                                       [[0, 0], [0, 1], [0, 0]],
                                       [[1, 0], [1, 1], [1, 0]],
                                       [[1, 0], [1, 1], [1, 0]]]

    def get_planning_problem(self, num_agents):
        global_model_ids = self.global_model_ids
        agent_skeleton_l = [self.agent_skeleton_options[i] for i in range(num_agents)]

        # Get random starts and goals as if all start tiles and goal tiles are in highways.
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(num_agents=num_agents,
                                                                               env_class=EnvHighways2D,
                                                                               tensor_args=params.tensor_args,
                                                                               margin=0.2,
                                                                               obstacle_margin=0.2)

        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l


class EnvTestThreeByThreeRobotPlanarDiskRandom(MMDPlanningProblemConfig):
    def __init__(self):
        self.name = "EnvTestThreeByThree_RobotPlanarDisk_Random"
        self.start_state_pos_l, self.goal_state_pos_l = get_start_goal_pos_circle(10, radius=0.45)
        more_start_state_pos_l, more_goal_state_pos_l = get_start_goal_pos_circle(20, radius=0.65)
        self.start_state_pos_l += more_start_state_pos_l
        self.goal_state_pos_l += more_goal_state_pos_l

        self.global_model_ids = [['EnvEmptyNoWait2D-RobotPlanarDisk', 'EnvConveyor2D-RobotPlanarDisk', 'EnvDropRegion2D-RobotPlanarDisk'],
                                 ['EnvHighways2D-RobotPlanarDisk', 'EnvHighways2D-RobotPlanarDisk', 'EnvHighways2D-RobotPlanarDisk'],
                                 ['EnvConveyor2D-RobotPlanarDisk', 'EnvDropRegion2D-RobotPlanarDisk', 'EnvEmptyNoWait2D-RobotPlanarDisk']]
        self.agent_skeleton_options = [[[1, 1], [2, 1], [2, 2]],
                                       [[1, 2], [1, 1], [1, 2]],
                                       [[1, 1], [1, 2], [1, 1]],
                                       [[2, 2], [1, 2], [1, 1]],
                                       [[1, 0], [1, 1], [1, 2]],
                                       [[1, 1], [2, 1], [1, 1]],
                                       [[1, 0], [2, 0], [1, 0]],
                                       [[1, 1], [1, 0], [0, 0]],
                                       [[1, 1], [1, 2], [2, 2]],
                                       [[1, 2], [2, 2], [1, 2]],
                                       [[2, 2], [2, 1], [2, 2]],
                                       [[2, 2], [2, 1], [1, 1]],
                                       [[1, 2], [1, 1], [1, 0]],
                                       [[0, 0], [1, 0], [1, 1]],
                                       [[0, 0], [0, 1], [1, 1]],
                                       [[1, 0], [1, 1], [1, 0]],
                                       [[2, 2], [1, 2], [2, 2]],
                                       [[1, 1], [0, 1], [1, 1]],
                                       [[1, 1], [1, 0], [1, 1]],
                                       [[0, 0], [0, 1], [0, 0]],
                                       [[1, 2], [0, 2], [1, 2]],
                                       [[1, 0], [0, 0], [1, 0]],
                                       [[0, 0], [1, 0], [0, 0]],
                                       [[1, 1], [0, 1], [0, 0]]]

    def get_skeleton_options(self, optional_start_goal_coords, length=3, num_agents=3):
        # Return a list of paths starting from the optional_start_goal_coords and ending there too of length length.
        agent_skeleton_options = []
        for agent_id in range(num_agents):
            agent_skeleton_options.append([])
            for i in range(length):
                agent_skeleton_options[agent_id].append(optional_start_goal_coords[agent_id])

    def get_planning_problem(self, num_agents):

        # Get random starts and goals as if all start tiles and goal tiles are in highways.
        start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(num_agents=num_agents,
                                                                               env_class=EnvHighways2D,
                                                                               tensor_args=params.tensor_args,
                                                                               margin=0.2,
                                                                               obstacle_margin=0.2)

        global_model_ids = self.global_model_ids
        agent_skeleton_l = [self.agent_skeleton_options[i] for i in range(num_agents)]

        return start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l
