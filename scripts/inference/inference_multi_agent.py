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
# from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import os
import pickle
from datetime import datetime
import time
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List
import concurrent.futures

# Project imports.
from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraint
from mmd.models import TemporalUnet, UNET_DIM_MULTS
from mmd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mmd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mmd.trainer import get_dataset, get_model
from mmd.utils.loading import load_params_from_yaml
from torch_robotics.robots import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints, \
    compute_average_acceleration, compute_average_acceleration_from_pos_vel, compute_path_length_from_pos
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from mmd.planners.multi_agent import CBS, PrioritizedPlanning
from mmd.planners.single_agent import MPD, MPDEnsemble
from mmd.common.constraints import MultiPointConstraint, VertexConstraint, EdgeConstraint
from mmd.common.conflicts import VertexConflict, PointConflict, EdgeConflict
from mmd.common.trajectory_utils import smooth_trajs, densify_trajs
from mmd.common import compute_collision_intensity, is_multi_agent_start_goal_states_valid, global_pad_paths, \
    get_start_goal_pos_circle, get_state_pos_column, get_start_goal_pos_boundary, get_start_goal_pos_random_in_env
from mmd.common.pretty_print import *
from mmd.config.mmd_params import MMDParams as params
from mmd.common.experiments import MultiAgentPlanningSingleTrialConfig, MultiAgentPlanningSingleTrialResult, \
    get_result_dir_from_trial_config, TrialSuccessStatus
from torch_robotics.environments import *

allow_ops_in_compiled_graph()

TRAINED_MODELS_DIR = '../../data_trained_models/'
device = 'cuda'
device = get_torch_device(device)
tensor_args = {'device': device, 'dtype': torch.float32}


def run_multi_agent_trial(test_config: MultiAgentPlanningSingleTrialConfig):
    # ============================
    # Start time per agent.
    # ============================
    start_time_l = [i * test_config.stagger_start_time_dt for i in range(test_config.num_agents)]

    # ============================
    # Arguments for the high/low level planner.
    # ============================
    low_level_planner_model_args = {
        'planner_alg': 'mmd',
        'use_guide_on_extra_objects_only': params.use_guide_on_extra_objects_only,
        'n_samples': params.n_samples,
        'n_local_inference_noising_steps': params.n_local_inference_noising_steps,
        'n_local_inference_denoising_steps': params.n_local_inference_denoising_steps,
        'start_guide_steps_fraction': params.start_guide_steps_fraction,
        'n_guide_steps': params.n_guide_steps,
        'n_diffusion_steps_without_noise': params.n_diffusion_steps_without_noise,
        'weight_grad_cost_collision': params.weight_grad_cost_collision,
        'weight_grad_cost_smoothness': params.weight_grad_cost_smoothness,
        'weight_grad_cost_constraints': params.weight_grad_cost_constraints,
        'weight_grad_cost_soft_constraints': params.weight_grad_cost_soft_constraints,
        'factor_num_interpolated_points_for_collision': params.factor_num_interpolated_points_for_collision,
        'trajectory_duration': params.trajectory_duration,
        'device': params.device,
        'debug': params.debug,
        'seed': params.seed,
        'results_dir': params.results_dir,
        'trained_models_dir': TRAINED_MODELS_DIR,
    }
    high_level_planner_model_args = {
        'is_xcbs': True if test_config.multi_agent_planner_class in ["XECBS", "XCBS"] else False,
        'is_ecbs': True if test_config.multi_agent_planner_class in ["ECBS", "XECBS"] else False,
        'start_time_l': start_time_l,
        'runtime_limit': test_config.runtime_limit,
        'conflict_type_to_constraint_types': {PointConflict: {MultiPointConstraint}},
    }

    # ============================
    # Create a results directory.
    # ============================
    results_dir = get_result_dir_from_trial_config(test_config, test_config.time_str, test_config.trial_number)
    os.makedirs(results_dir, exist_ok=True)
    num_agents = test_config.num_agents

    # ============================
    # Get planning problem.
    # ============================
    # If want to get random starts and goals, then must do that after creating the reference task and robot.
    start_l = test_config.start_state_pos_l
    goal_l = test_config.goal_state_pos_l
    global_model_ids = test_config.global_model_ids
    agent_skeleton_l = test_config.agent_skeleton_l

    # ============================
    # Transforms and model tiles setup.
    # ============================
    # Create a reference planner from which we'll use the task and robot as the reference on in CBS.
    # Those are used for collision checking and visualization. This has a skeleton of all tiles.
    reference_agent_skeleton = [[r, c] for r in range(len(global_model_ids))
                                for c in range(len(global_model_ids[0]))]

    # ============================
    # Transforms from tiles to global frame.
    # ============================
    tile_width = 2.0
    tile_height = 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **tensor_args)
                                for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]

    # ============================
    # Parse the single agent planner class name.
    # ============================
    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_class = MPD
    elif test_config.single_agent_planner_class == "MPDEnsemble":
        low_level_planner_class = MPDEnsemble
    else:
        raise ValueError(f'Unknown single agent planner class: {test_config.single_agent_planner_class}')

    # ============================
    # Create reference agent planner.
    # ============================
    # And for the reference skeleton.
    reference_task = None
    reference_robot = None
    reference_agent_transforms = {}
    reference_agent_model_ids = {}
    for skeleton_step in range(len(reference_agent_skeleton)):
        skeleton_model_coord = reference_agent_skeleton[skeleton_step]
        reference_agent_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
        reference_agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
    reference_agent_model_ids = [reference_agent_model_ids[i] for i in range(len(reference_agent_model_ids))]
    # Create the reference low level planner.
    print("Creating reference agent stuff.")
    low_level_planner_model_args['start_state_pos'] = torch.tensor([0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['goal_state_pos'] = torch.tensor([-0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['model_ids'] = reference_agent_model_ids  # This matters.
    low_level_planner_model_args['transforms'] = reference_agent_transforms  # This matters.

    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_model_args['model_id'] = reference_agent_model_ids[0]

    reference_low_level_planner = low_level_planner_class(**low_level_planner_model_args)
    reference_task = reference_low_level_planner.task
    reference_robot = reference_low_level_planner.robot

    # ============================
    # Run trial.
    # ============================
    exp_name = f'mmd_single_trial'

    # Transform starts and goals to the global frame. Right now they are in the local tile frames.
    start_l = [start_l[i] + global_model_transforms[agent_skeleton_l[i][0][0]][agent_skeleton_l[i][0][1]]
               for i in range(num_agents)]
    goal_l = [goal_l[i] + global_model_transforms[agent_skeleton_l[i][-1][0]][agent_skeleton_l[i][-1][1]]
              for i in range(num_agents)]

    # ============================
    # Create global transforms for each agent's skeleton.
    # ============================
    # Each agent has a dict entry. Each entry is a dict with the skeleton steps (0, 1, 2, ...), mapping to the
    # model transform.
    agent_model_transforms_l = []
    agent_model_ids_l = []
    for agent_id in range(num_agents):
        agent_model_transforms = {}
        agent_model_ids = {}
        for skeleton_step in range(len(agent_skeleton_l[agent_id])):
            skeleton_model_coord = agent_skeleton_l[agent_id][skeleton_step]
            agent_model_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
                skeleton_model_coord[1]]
            agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][skeleton_model_coord[1]]
        agent_model_transforms_l.append(agent_model_transforms)
        agent_model_ids_l.append(agent_model_ids)
    # Change the dict of the model ids to a list sorted by the skeleton steps.
    agent_model_ids_l = [[agent_model_ids_l[i][j] for j in range(len(agent_model_ids_l[i]))] for i in
                         range(num_agents)]

    # ============================
    # Create the low level planners.
    # ============================
    planners_creation_start_time = time.time()
    low_level_planner_l = []
    for i in range(num_agents):
        low_level_planner_model_args_i = low_level_planner_model_args.copy()
        low_level_planner_model_args_i['start_state_pos'] = start_l[i]
        low_level_planner_model_args_i['goal_state_pos'] = goal_l[i]
        low_level_planner_model_args_i['model_ids'] = agent_model_ids_l[i]
        low_level_planner_model_args_i['transforms'] = agent_model_transforms_l[i]
        if test_config.single_agent_planner_class == "MPD":
            # Set the model_id to the first one.
            low_level_planner_model_args_i['model_id'] = agent_model_ids_l[i][0]
        low_level_planner_l.append(low_level_planner_class(**low_level_planner_model_args_i))
    print('Planners creation time:', time.time() - planners_creation_start_time)
    print("\n\n\n\n")

    # ============================
    # Create the multi agent planner.
    # ============================
    if (test_config.multi_agent_planner_class in ["XECBS", "ECBS", "XCBS", "CBS"]):
        multi_agent_planner_class = CBS
    elif test_config.multi_agent_planner_class == "PP":
        multi_agent_planner_class = PrioritizedPlanning
    else:
        raise ValueError(f'Unknown multi agent planner class: {test_config.multi_agent_planner_class}')
    planner = multi_agent_planner_class(low_level_planner_l,
                                        start_l,
                                        goal_l,
                                        reference_task=reference_task,
                                        reference_robot=reference_robot,
                                        **high_level_planner_model_args)
    # ============================
    # Plan.
    # ============================
    startt = time.time()
    paths_l, num_ct_expansions, trial_success_status, num_collisions_in_solution = \
        planner.plan(runtime_limit=test_config.runtime_limit)
    planning_time = time.time() - startt
    # Print planning times.
    print(GREEN, 'Planning times:', planning_time, RESET)

    # ============================
    # Gather stats.
    # ============================
    single_trial_result = MultiAgentPlanningSingleTrialResult()
    # The associated experiment config.
    single_trial_result.trial_config = test_config
    # The planning problem.
    single_trial_result.start_state_pos_l = [start_l[i].cpu().numpy().tolist() for i in range(num_agents)]
    single_trial_result.goal_state_pos_l = [goal_l[i].cpu().numpy().tolist() for i in range(num_agents)]
    single_trial_result.global_model_ids = global_model_ids
    single_trial_result.agent_skeleton_l = agent_skeleton_l
    # The agent paths. Each entry is of shape (H, 4).
    single_trial_result.agent_path_l = paths_l
    # Success.
    single_trial_result.success_status = trial_success_status
    # Number of collisions in the solution.
    single_trial_result.num_collisions_in_solution = num_collisions_in_solution
    # Planning time.
    single_trial_result.planning_time = planning_time

    # Number of agent pairs in collision.
    if len(paths_l) > 0 and trial_success_status:
        # This assumes all paths in the solution are of the same length.
        for t in range(len(paths_l[0])):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    if torch.norm(paths_l[i][t, :2] - paths_l[j][t, :2]) < 2.0 * params.robot_planar_disk_radius:
                        # The above should be reference_robot.radius.
                        print(RED, 'Collision in solution:', i, j, t, paths_l[i][t, :2], paths_l[j][t, :2], RESET)
                        single_trial_result.num_collisions_in_solution += 1
        if single_trial_result.num_collisions_in_solution > 0:
            single_trial_result.success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS

    # If not successful, return here.
    if trial_success_status:
        # Our metric for determining how well a path is adhering to the data.
        # Computed by the environment. If it is a single map, the score is the adherence there.
        # If it is a multi-tile map, the score is the average adherence over all tiles.
        single_trial_result.data_adherence = 0.0
        for agent_id in range(num_agents):
            agent_data_adherence = 0.0
            for skeleton_step, agent_model_id in enumerate(agent_model_ids_l[agent_id]):
                agent_model_transform = agent_model_transforms_l[agent_id][skeleton_step]
                agent_start_time = start_time_l[agent_id]
                single_tile_traj_len = params.horizon
                agent_path_in_model_frame = (paths_l[agent_id].clone()[
                                             agent_start_time + skeleton_step * single_tile_traj_len:
                                             agent_start_time + (skeleton_step + 1) * single_tile_traj_len, :2] -
                                             agent_model_transform)
                model_env_name = agent_model_id.split('-')[0]
                kwargs = {'tensor_args': tensor_args}
                env_object = eval(model_env_name)(**kwargs)
                agent_data_adherence += env_object.compute_traj_data_adherence(agent_path_in_model_frame)
            agent_data_adherence /= len(agent_model_ids_l[agent_id])
            single_trial_result.data_adherence += agent_data_adherence
        single_trial_result.data_adherence /= num_agents
        # CT nodes expanded.
        single_trial_result.num_ct_expansions = num_ct_expansions
        # Path length. Hack for experiments.
        single_trial_result.path_length_per_agent = 0.0
        for agent_id in range(num_agents):
            # agent_path_pos = low_level_planner_l[agent_id].robot.get_position(paths_l[agent_id]).unsqueeze(0)
            agent_path_pos = paths_l[agent_id][:, :2].unsqueeze(0)
            single_trial_result.path_length_per_agent += compute_path_length_from_pos(agent_path_pos).item()
        single_trial_result.path_length_per_agent /= num_agents
        # Path smoothness.
        single_trial_result.mean_path_acceleration_per_agent = 0.0
        for agent_id in range(num_agents):
            # agent_path_pos = low_level_planner_l[agent_id].robot.get_position(paths_l[agent_id]).unsqueeze(0)
            # agent_path_vel = low_level_planner_l[agent_id].robot.get_velocity(paths_l[agent_id]).unsqueeze(0)
            agent_path_pos = paths_l[agent_id][:, :2].unsqueeze(0)
            agent_path_vel = paths_l[agent_id][:, 2:].unsqueeze(0)
            if agent_path_vel.shape[-1] == 0:
                agent_path_vel = low_level_planner_l[agent_id].robot.get_velocity(paths_l[agent_id]).unsqueeze(0)

            single_trial_result.mean_path_acceleration_per_agent += (
                compute_average_acceleration_from_pos_vel(agent_path_pos, agent_path_vel).item())
        single_trial_result.mean_path_acceleration_per_agent /= num_agents
    # ============================
    # Save the results and config.
    # ============================
    print(GREEN, single_trial_result, RESET)
    results_dir_uri = f'file://{os.path.abspath(results_dir)}'
    print('Results dir:', results_dir_uri)
    single_trial_result.save(results_dir)
    test_config.save(results_dir)

    # ============================
    # Render.
    # ============================
    if trial_success_status and len(paths_l) > 0:
        planner.render_paths(paths_l,
                             output_fpath=os.path.join(results_dir, f'{exp_name}.gif'),
                             animation_duration=0,
                             plot_trajs=True,
                             show_robot_in_image=True)
        if test_config.render_animation:
            paths_l = densify_trajs(paths_l, 1)  # <------ Larger numbers produce nicer animations. But take longer to make too.
            planner.render_paths(paths_l,
                                 output_fpath=os.path.join(results_dir, f'{exp_name}.gif'),
                                 plot_trajs=True,
                                 animation_duration=10)


if __name__ == '__main__':
    test_config_single_tile = MultiAgentPlanningSingleTrialConfig()
    test_config_single_tile.num_agents = 3
    test_config_single_tile.instance_name = "test"
    test_config_single_tile.multi_agent_planner_class = "XECBS"  # Or "ECBS" or "XCBS" or "CBS" or "PP".
    test_config_single_tile.single_agent_planner_class = "MPDEnsemble"  # Or "MPD"
    test_config_single_tile.stagger_start_time_dt = 0
    test_config_single_tile.runtime_limit = 60 * 3  # 3 minutes.
    test_config_single_tile.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_config_single_tile.render_animation = True  # Change the `densify_trajs` call above to create nicer animations.

    example_type = "single_tile"
    # example_type = "multi_tile"
    # ============================
    # Single tile.
    # ============================
    if example_type == "single_tile":
        # Choose the model to use. A model is for a map/robot combination.
        # test_config_single_tile.global_model_ids = [['EnvEmpty2D-RobotPlanarDisk']]
        test_config_single_tile.global_model_ids = [['EnvEmptyNoWait2D-RobotPlanarDisk']]
        # test_config_single_tile.global_model_ids = [['EnvConveyor2D-RobotPlanarDisk']]
        # test_config_single_tile.global_model_ids = [['EnvHighways2D-RobotPlanarDisk']]
        # test_config_single_tile.global_model_ids = [['EnvDropRegion2D-RobotPlanarDisk']]

        # Choose starts and goals.
        test_config_single_tile.agent_skeleton_l = [[[0, 0]]] * test_config_single_tile.num_agents
        torch.random.manual_seed(10)
        test_config_single_tile.start_state_pos_l, test_config_single_tile.goal_state_pos_l = \
        get_start_goal_pos_circle(test_config_single_tile.num_agents, 0.8)
        # Another option is to get random starts and goals.
        # get_start_goal_pos_random_in_env(test_config_single_tile.num_agents,
        #                                  EnvDropRegion2D,
        #                                  tensor_args,
        #                                  margin=0.2,
        #                                  obstacle_margin=0.11)
        # A third option is to get starts and goals in a "boundary" formation.
        # get_start_goal_pos_boundary(test_config_single_tile.num_agents, 0.85)
        # And a final option is to hard-code starts and goals.
        # (torch.tensor([[-0.8, 0], [0.8, -0]], **tensor_args),
        #  torch.tensor([[0.8, -0], [-0.8, 0]], **tensor_args))
        print("Starts:", test_config_single_tile.start_state_pos_l)
        print("Goals:", test_config_single_tile.goal_state_pos_l)

        run_multi_agent_trial(test_config_single_tile)
        print(GREEN, 'OK.', RESET)

    # ============================
    # Multiple tiles example.
    # ============================
    if example_type == "multi_tile":
        test_config_multiple_tiles = test_config_single_tile
        test_config_multiple_tiles.num_agents = 4
        test_config_multiple_tiles.stagger_start_time_dt = 5
        test_config_multiple_tiles.global_model_ids = \
            [['EnvEmptyNoWait2D-RobotPlanarDisk', 'EnvEmptyNoWait2D-RobotPlanarDisk']]

        test_config_multiple_tiles.agent_skeleton_l = [[[0, 0], [0, 1]],
                                                       [[0, 1], [0, 0]],
                                                       [[0, 0], [0, 1]],
                                                       [[0, 1], [0, 0]]]
        test_config_multiple_tiles.start_state_pos_l, test_config_multiple_tiles.goal_state_pos_l = \
            (torch.tensor([[0, 0.8], [0, 0.3], [0, -0.3], [0, -0.8]], **tensor_args),
             torch.tensor([[0, -0.8], [0, -0.3], [0, 0.3], [0, 0.8]], **tensor_args))
        print(test_config_multiple_tiles.start_state_pos_l)
        test_config_multiple_tiles.multi_agent_planner_class = "XECBS"
        run_multi_agent_trial(test_config_multiple_tiles)
        print(GREEN, 'OK.', RESET)
