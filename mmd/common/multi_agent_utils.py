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
# Standard imports.
import torch
from typing import List
from copy import deepcopy
import numpy as np
# Project imports.


def is_multi_agent_state_valid(reference_robot,
                               reference_task,
                               state_pos_l: List[torch.Tensor]
                               ) -> bool:
    """
    Check if a state is valid.
    :param reference_robot: Reference robot.
    :param reference_task: Reference task.
    :param state_pos_l: State position list of tensors.
    :return: True if the state is valid, False otherwise.
    """
    state_pos = torch.stack(state_pos_l)
    collision_matrix, _ = reference_robot.check_rr_collisions(state_pos)
    if torch.any(collision_matrix):
        return False
    world_collisions = reference_task.compute_collision(state_pos)
    if torch.any(world_collisions):
        return False
    return True


def is_multi_agent_start_goal_states_valid(reference_robot,
                                           reference_task,
                                           start_state_pos_l: List[torch.Tensor],
                                           goal_state_pos_l: List[torch.Tensor],
                                           is_enforce_min_dist: bool = True
                                           ) -> bool:
    if is_enforce_min_dist:
        for i in range(len(start_state_pos_l)):
            for j in range(i + 1, len(goal_state_pos_l)):
                # Start-start.
                if torch.norm(start_state_pos_l[i] - start_state_pos_l[j]) < 0.15:
                    print('Start-start failed with i:', i, 'j:', j, " distance:",
                          torch.norm(start_state_pos_l[i] - start_state_pos_l[j]))
                    return False
                # Goal-goal.
                if torch.norm(goal_state_pos_l[i] - goal_state_pos_l[j]) < 0.15:
                    print('Goal-goal failed with i:', i, 'j:', j, " distance:",
                          torch.norm(goal_state_pos_l[i] - goal_state_pos_l[j]))
                    return False

    # Check for collisions with the world and between robots.
    starts_collision_matrix, _ = reference_robot.check_rr_collisions(torch.stack(start_state_pos_l))
    if torch.any(starts_collision_matrix):
        print('Start states are in collision.')
        print("The collision matrix is", starts_collision_matrix)
        return False
    goals_collision_matrix, _ = reference_robot.check_rr_collisions(torch.stack(goal_state_pos_l))
    if torch.any(goals_collision_matrix):
        print('Goal states are in collision.')
        print("The collision matrix is", goals_collision_matrix)
        return False
    starts_world_collisions = reference_task.compute_collision(torch.stack(start_state_pos_l))
    if torch.any(starts_world_collisions):
        print('Start states are in collision with the world.')
        print("The collision matrix is", starts_world_collisions)
        return False
    goals_world_collisions = reference_task.compute_collision(torch.stack(goal_state_pos_l))
    if torch.any(goals_world_collisions):
        print('Goal states are in collision with the world.')
        print("The collision matrix is", goals_world_collisions)
        return False
    return True


def compute_collision_intensity(trajs_l, reference_robot, reference_task):
    """
    Compute the collision intensity for the multi-agent trajectory.
    Intensity is defined as the fraction of trajectory timesteps that are in collision.
    :param trajs: List of trajectories. Each is a tensor of shape (H, n_dims).
    """
    n_agents = len(trajs_l)
    n_timesteps = trajs_l[0].shape[0]
    # Make sure all trajectories are of the same length.
    assert all(traj.shape[0] == n_timesteps for traj in trajs_l)
    collision_intensity = torch.zeros(n_timesteps)
    for t in range(n_timesteps):
        state_pos_l = [trajs_l[agent_id][t] for agent_id in range(n_agents)]
        step_valid = is_multi_agent_state_valid(reference_robot, reference_task, state_pos_l)
        if not step_valid:
            print("Collision at timestep", t)
            collision_intensity[t] = 1
    print("num steps colliding", collision_intensity.sum())
    collision_intensity = collision_intensity.sum() / n_timesteps
    # collision_intensity /= n_agents * (n_agents - 1) / 2
    return collision_intensity


def global_pad_paths(path_l: List[torch.Tensor],
                     start_time_l: List[int],
                     ) -> List[torch.Tensor]:
    # Agents starting at time later than 0 will get repeated start state.
    # Agents ending at time earlier than the max_t will get repeated last state.
    # Find the maximum t among all paths.
    path_l = deepcopy(path_l)
    # Check if there is anything there.
    if len(path_l) == 0:
        return path_l
    max_t = max([len(path) + start_time_l[agent_id] for agent_id, path in enumerate(path_l)])
    for agent_id, agent_path in enumerate(path_l):
        if len(agent_path) + start_time_l[agent_id] < max_t:
            # Repeat the last state.
            agent_path = torch.cat(
                [agent_path, agent_path[-1].repeat(max_t - len(agent_path) - start_time_l[agent_id], 1)]
            )
        if start_time_l[agent_id] > 0:
            # Repeat the start state.
            agent_path = torch.cat(
                [agent_path[0].repeat(start_time_l[agent_id], 1), agent_path]
            )
        path_l[agent_id] = agent_path
    return path_l


def get_start_goal_pos_circle(num_agents: int, radius=0.8):
    # These are all in the local tile frame.
    start_l = [torch.tensor([radius * np.cos(2 * torch.pi * i / num_agents),
                             radius * np.sin(2 * torch.pi * i / num_agents)],
                            dtype=torch.float32, device='cuda') for i in range(num_agents)]
    goal_l = [torch.tensor([radius * np.cos(2 * torch.pi * i / num_agents + torch.pi),
                            radius * np.sin(2 * torch.pi * i / num_agents + torch.pi)],
                           dtype=torch.float32, device='cuda') for i in range(num_agents)]
    return start_l, goal_l


def get_start_goal_pos_boundary(num_agents: int, dist=0.87):
    # These are all in the local tile frame.
    start_l = [torch.tensor([0.8 * np.cos(2 * torch.pi * i / num_agents),
                             0.8 * np.sin(2 * torch.pi * i / num_agents)],
                            dtype=torch.float32, device='cuda') for i in range(num_agents)]

    # Snap the abs max x or y value to either -1 or 1.
    for i in range(num_agents):
        if abs(start_l[i][0]) > abs(start_l[i][1]):
            start_l[i][0] = torch.sign(start_l[i][0]) * dist
        else:
            start_l[i][1] = torch.sign(start_l[i][1]) * dist

    goal_l = [torch.tensor([start_l[i][0] if abs(start_l[i][0]) < abs(start_l[i][1]) else -start_l[i][0],
                            start_l[i][1] if abs(start_l[i][1]) < abs(start_l[i][0]) else -start_l[i][1]],
                           dtype=torch.float32, device='cuda') for i in range(num_agents)]
    return start_l, goal_l


def get_state_pos_column(num_agents: int, x_pos: float):
    # These are all in the local tile frame.
    state_l = [torch.tensor([x_pos, 0.8 * (1 - 2 * i / num_agents)],
                            dtype=torch.float32, device='cuda') for i in range(num_agents)]
    return state_l


def get_start_goal_pos_random_in_env(num_agents,
                                     env_class,
                                     tensor_args,
                                     margin=0.15,
                                     obstacle_margin=0.16):  # 0.08
    # In this map, agents can be place anywhere with x y in [-0.9, 0.9]. As long as they are not too close to
    # each other.
    start_state_pos_l = []
    goal_state_pos_l = []

    # Get the obstacles in this map.
    env = env_class(tensor_args=tensor_args)
    # Get a distance field.
    env_sdf = env.grid_map_sdf_obj_fixed

    # Get the starts. Get all of them together and then check if they are too close.
    for i in [0, 1]:
        # Start building the random state.
        random_state = None
        while True:
            random_state = torch.rand(1, 2) * 1.9 - 0.95
            random_state = random_state.to(**tensor_args)
            # Check if the state is not in an obstacle.
            if torch.all(env_sdf(random_state) > obstacle_margin):
                break
        # Now add more point. For each point, check if it is not too close to the previous points and not in an
        # obstacle.
        state_b = random_state
        for _ in range(num_agents - 1):
            while True:
                new_state = torch.rand(1, 2) * 1.9 - 0.95
                new_state = new_state.to(**tensor_args)
                pairwise_distances = torch.sqrt(torch.sum((new_state - state_b) ** 2, dim=1))
                # Check if the state is not in an obstacle.
                if torch.all(env_sdf(new_state) > obstacle_margin) and torch.all(pairwise_distances > margin):
                    break
            state_b = torch.cat([state_b, new_state], dim=0)
        if i == 0:
            start_state_pos_l = [s for s in state_b]
        else:
            goal_state_pos_l = [s for s in state_b]

    return start_state_pos_l, goal_state_pos_l
