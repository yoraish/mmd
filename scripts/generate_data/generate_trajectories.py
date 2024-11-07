"""
Adapted from https://github.com/jacarvalho/mpd-public
"""
import os
import pickle
import time
from pathlib import Path

import torch
import yaml
from matplotlib import pyplot as plt
from typing import List

# Project includes.
from experiment_launcher import single_experiment_yaml, run_experiment
from experiment_launcher.utils import fix_random_seed
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.costs.cost_functions import *
from mp_baselines.planners.hybrid_planner import HybridPlanner
from mp_baselines.planners.multi_sample_based_planner import MultiSampleBasedPlanner
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.identity_planner import IdentityPlanner
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from mmd.common.trajectory_utils import densify_trajs


def generate_collision_free_trajectories(
    env_id,
    robot_id,
    num_trajectories_per_context,
    results_dir,
    threshold_start_goal_pos=0.5,
    obstacle_cutoff_margin=0.03,
    n_tries=1000,
    rrt_max_time=300,
    gpmp_opt_iters=500,
    n_support_points=64,
    duration=5.0,
    tensor_args=None,
    debug=False,
    is_start_goal_near_limits=False,
):
    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    print("hybrid planner")
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    # Task
    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        # Ask the environment if this start and goal are valid for data generation.
        if not env.is_start_goal_valid_for_data_gen(robot, start_state_pos, goal_state_pos):
            print(f"Invalid start and goal for data generation: {start_state_pos}, {goal_state_pos}")
            continue

        if is_start_goal_near_limits:
            dim = torch.randint(0, robot.q_dim, (1,)).item()
            random_number = torch.rand(1, **tensor_args)
            dim_range = robot.q_max[dim] - robot.q_min[dim]
            if random_number < 1/robot.q_dim:
                start_state_pos[dim] = robot.q_min[dim] + dim_range * 0.1
            else:
                start_state_pos[dim] = robot.q_max[dim] - dim_range * 0.1
            if random_number < 1/robot.q_dim:
                goal_state_pos[dim] = robot.q_max[dim] - dim_range * 0.1
            else:
                goal_state_pos[dim] = robot.q_min[dim] + dim_range * 0.1

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    n_trajectories = num_trajectories_per_context

    # Get the skill sequence conditioned on the start and goal.
    skill_pos_sequence_l = env.get_skill_pos_seq_l(robot, start_pos=start_state_pos, goal_pos=goal_state_pos)

    # -------------------------------- Hybrid Planner ---------------------------------
    # Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    rrt_connect_default_params_env['max_time'] = rrt_max_time

    # Two options here. Either ask for an RRT-Conect path from random start to goal. Or ask for RRT-Connect path from
    # start to beginning of skill sequence, and then from end of skill sequence to goal.
    # This is to ensure that the skill is included as-is.
    pre_optimization_planners = []  # Their solutions will be concatenated.
    if not skill_pos_sequence_l:
        rrt_connect_params = dict(
            **rrt_connect_default_params_env,
            task=task,
            start_state_pos=start_state_pos,
            goal_state_pos=goal_state_pos,
            tensor_args=tensor_args,
        )
        sample_based_planner_base = RRTConnect(**rrt_connect_params)
        sample_based_planner = MultiSampleBasedPlanner(
            sample_based_planner_base,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        pre_optimization_planners = [sample_based_planner]
    else:
        # Choose random skill index to choose. There may be multiple demonstrations of a skill so choose one.
        rand_ix_skill = torch.randint(0, len(skill_pos_sequence_l), (1,)).item()
        rrt_start_to_skill = RRTStar(
            **rrt_connect_default_params_env,
            task=task,
            start_state_pos=start_state_pos,
            goal_state_pos=skill_pos_sequence_l[rand_ix_skill][0],
            tensor_args=tensor_args,
        )
        skill = IdentityPlanner(skill_pos_sequence_l[rand_ix_skill], tensor_args=tensor_args)
        rrt_skill_to_goal = RRTStar(
            **rrt_connect_default_params_env,
            task=task,
            start_state_pos=skill_pos_sequence_l[rand_ix_skill][-1],
            goal_state_pos=goal_state_pos,
            tensor_args=tensor_args,
        )
        planner_start_to_skill = MultiSampleBasedPlanner(
            rrt_start_to_skill,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        planner_skill = MultiSampleBasedPlanner(
            skill,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        planner_skill_to_goal = MultiSampleBasedPlanner(
            rrt_skill_to_goal,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        pre_optimization_planners = [planner_start_to_skill, planner_skill, planner_skill_to_goal]

    # Optimization-based planner
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=n_trajectories,
        start_state=start_state_pos,
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    ###############
    # Hybrid planner
    planner = HybridPlanner(
        pre_optimization_planners,
        opt_based_planner,
        tensor_args=tensor_args
    )

    # Optimize
    trajs_iters = planner.optimize(debug=debug, print_times=True, return_iterations=True)
    trajs_last_iter = trajs_iters[-1]

    # -------------------------------- Save trajectories ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_last_iter)*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_last_iter)*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_last_iter)}')

    # save
    torch.cuda.empty_cache()
    trajs_last_iter_coll, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)
    if trajs_last_iter_coll is None:
        trajs_last_iter_coll = torch.empty(0)
    torch.save(trajs_last_iter_coll, os.path.join(results_dir, f'trajs-collision.pt'))
    if trajs_last_iter_free is None:
        trajs_last_iter_free = torch.empty(0)
    torch.save(trajs_last_iter_free, os.path.join(results_dir, f'trajs-free.pt'))

    # save results data dict
    trajs_iters_coll, trajs_iters_free = task.get_trajs_collision_and_free(trajs_iters[-1])
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': planner_params['dt'],
        'trajs_iters_coll': trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        'trajs_iters_free': trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(os.path.join(results_dir, f'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(task=task)

    trajs = trajs_last_iter_free
    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories.png'), dpi=100)
    plt.close(fig)

    # Plot 2d trajectories if the environment is 2D.
    # if "Empty" in env_id:
    print(f'Plotting 2D trajectories')
    fig, axs = planner_visualizer.render_robot_trajectories(
        trajs=trajs
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories_2d.png'), dpi=100)
    plt.close(fig)

    # Plot 2d trajectories if the environment is 2D.
    # if "Empty" in env_id:
    print(f'Plotting 2D trajectories')
    fig, axs = planner_visualizer.render_robot_trajectories(
        trajs=trajs
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories_2d.png'), dpi=100)
    plt.close(fig)

    # Create animation. Do this with CBS. Yes I know, it's weird.
    # Densify the paths for visualization.
    trajs_l = [traj for traj in trajs]
    trajs_dense_l = densify_trajs(trajs_l, 1)
    # Add batch dimension to all paths.
    trajs_dense_l = [traj.unsqueeze(0) for traj in trajs_dense_l]

    base_file_name = Path(os.path.basename(__file__)).stem
    output_fpath = os.path.join(results_dir, f'{base_file_name}-robot-traj.gif')
    # Render the paths.
    print('Rendering paths and saving to:', output_fpath)
    # Print a link path to terminal that can be clicked to get there.
    file_uri = f'file://{os.path.realpath(output_fpath)}'
    print(f'Click to open output  directory:file://{os.path.realpath(results_dir)}')

    # print(f'Click to open GIF: {file_uri}')
    # planner_visualizer.animate_multi_robot_trajectories(
    #     trajs_l=trajs_dense_l,
    #     start_state_l=[pos_trajs[i][0] for i in range(len(trajs_dense_l))],
    #     goal_state_l=[pos_trajs[i][-1] for i in range(len(trajs_dense_l))],
    #     plot_trajs=True,
    #     video_filepath=output_fpath,
    #     n_frames=max((2, trajs_dense_l[0].shape[1])),
    #     # n_frames=pos_trajs_iters[-1].shape[1],
    #     anim_time=15.0,
    #     constraints=None,
    #     colors=[plt.cm.tab10(i) for i in range(len(trajs_dense_l))],
    # )

    num_trajectories_coll, num_trajectories_free = len(trajs_last_iter_coll), len(trajs_last_iter_free)
    return num_trajectories_coll, num_trajectories_free


def generate_skill_only_trajectories(
    env_id,
    robot_id,
    num_trajectories_per_context,
    results_dir,
    threshold_start_goal_pos=1.0,
    obstacle_cutoff_margin=0.03,
    n_tries=1000,
    rrt_max_time=300,
    gpmp_opt_iters=500,
    n_support_points=64,
    duration=5.0,
    tensor_args=None,
    debug=False,
    # skill_pos_sequence_l: List[torch.Tensor] = None,  # Each entry is a (n, 2) tensor of positions.
    is_start_goal_near_limits=False,
):
    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    print("hybrid planner")
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    # Task
    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]  # Shape (robot.q_dim,)
        goal_state_pos = q_free[1]  # Shape (robot.q_dim,)

        # Ask the environment if this start and goal are valid for data generation.
        if not env.is_start_goal_valid_for_data_gen(robot, start_state_pos, goal_state_pos):
            print(f"Invalid start and goal for data generation: {start_state_pos}, {goal_state_pos}")
            continue

        if is_start_goal_near_limits:
            dim = torch.randint(0, robot.q_dim, (1,)).item()
            random_number = torch.rand(1, **tensor_args)
            dim_range = robot.q_max[dim] - robot.q_min[dim]
            if random_number < 1/robot.q_dim:
                start_state_pos[dim] = robot.q_min[dim] + dim_range * 0.15
            else:
                start_state_pos[dim] = robot.q_max[dim] - dim_range * 0.15
            if random_number < 1/robot.q_dim:
                goal_state_pos[dim] = robot.q_max[dim] - dim_range * 0.15
            else:
                goal_state_pos[dim] = robot.q_min[dim] + dim_range * 0.15

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    n_trajectories = num_trajectories_per_context

    # Get the skill sequence conditioned on the start and goal.
    skill_pos_sequence_l = env.get_skill_pos_seq_l(robot, start_pos=start_state_pos, goal_pos=goal_state_pos)

    # -------------------------------- Hybrid Planner ---------------------------------
    # Sample-based planner
    rrt_connect_default_params_env = env.get_rrt_connect_params(robot=robot)
    rrt_connect_default_params_env['max_time'] = rrt_max_time

    # Two options here. Either ask for an RRT-Conect path from random start to goal. Or ask for RRT-Connect path from
    # start to beginning of skill sequence, and then from end of skill sequence to goal.
    # This is to ensure that the skill is included as-is.
    pre_optimization_planners = []  # Their solutions will be concatenated.
    if not skill_pos_sequence_l:
        rrt_connect_params = dict(
            **rrt_connect_default_params_env,
            task=task,
            start_state_pos=start_state_pos,
            goal_state_pos=goal_state_pos,
            tensor_args=tensor_args,
        )
        sample_based_planner_base = RRTConnect(**rrt_connect_params)
        sample_based_planner = MultiSampleBasedPlanner(
            sample_based_planner_base,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        pre_optimization_planners = [sample_based_planner]
    else:
        # Choose random skill index to choose. There may be multiple demonstrations of a skill so choose one.
        rand_ix_skill = torch.randint(0, len(skill_pos_sequence_l), (1,)).item()
        rrt_start_to_skill = RRTConnect(
            **rrt_connect_default_params_env,
            task=task,
            start_state_pos=start_state_pos,
            goal_state_pos=skill_pos_sequence_l[rand_ix_skill][0],
            tensor_args=tensor_args,
        )
        skill = IdentityPlanner(skill_pos_sequence_l[rand_ix_skill], tensor_args=tensor_args)
        rrt_skill_to_goal = RRTConnect(
            **rrt_connect_default_params_env,
            task=task,
            start_state_pos=skill_pos_sequence_l[rand_ix_skill][-1],
            goal_state_pos=goal_state_pos,
            tensor_args=tensor_args,
        )
        planner_start_to_skill = MultiSampleBasedPlanner(
            rrt_start_to_skill,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        planner_skill = MultiSampleBasedPlanner(
            skill,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        planner_skill_to_goal = MultiSampleBasedPlanner(
            rrt_skill_to_goal,
            n_trajectories=n_trajectories,
            max_processes=-1,
            optimize_sequentially=True
        )
        pre_optimization_planners = [planner_start_to_skill, planner_skill, planner_skill_to_goal]

    # Optimization-based planner
    gpmp_default_params_env = env.get_gpmp2_params(robot=robot)
    gpmp_default_params_env['opt_iters'] = gpmp_opt_iters
    gpmp_default_params_env['n_support_points'] = n_support_points
    gpmp_default_params_env['dt'] = duration / n_support_points

    planner_params = dict(
        **gpmp_default_params_env,
        robot=robot,
        n_dof=robot.q_dim,
        num_particles_per_goal=n_trajectories,
        start_state=start_state_pos,
        multi_goal_states=goal_state_pos.unsqueeze(0),  # add batch dim for interface,
        collision_fields=task.get_collision_fields(),
        tensor_args=tensor_args,
    )
    opt_based_planner = GPMP2(**planner_params)

    ###############
    # Hybrid planner
    planner = HybridPlanner(
        pre_optimization_planners,
        opt_based_planner,
        tensor_args=tensor_args
    )

    # Optimize
    trajs_iters = planner.optimize(debug=debug, print_times=True, return_iterations=True)
    trajs_last_iter = trajs_iters[-1]

    # -------------------------------- Save trajectories ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_last_iter)*100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_last_iter)*100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_last_iter)}')

    # save
    torch.cuda.empty_cache()
    trajs_last_iter_coll, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)
    if trajs_last_iter_coll is None:
        trajs_last_iter_coll = torch.empty(0)
    torch.save(trajs_last_iter_coll, os.path.join(results_dir, f'trajs-collision.pt'))
    if trajs_last_iter_free is None:
        trajs_last_iter_free = torch.empty(0)
    torch.save(trajs_last_iter_free, os.path.join(results_dir, f'trajs-free.pt'))

    # save results data dict
    trajs_iters_coll, trajs_iters_free = task.get_trajs_collision_and_free(trajs_iters[-1])
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': planner_params['dt'],
        'trajs_iters_coll': trajs_iters_coll.unsqueeze(0) if trajs_iters_coll is not None else None,
        'trajs_iters_free': trajs_iters_free.unsqueeze(0) if trajs_iters_free is not None else None,
    }

    with open(os.path.join(results_dir, f'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(task=task)

    trajs = trajs_last_iter_free
    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories.png'), dpi=100)
    plt.close(fig)

    # Plot 2d trajectories if the environment is 2D.
    # if "Empty" in env_id:
    print(f'Plotting 2D trajectories')
    fig, axs = planner_visualizer.render_robot_trajectories(
        trajs=trajs
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories_2d.png'), dpi=100)
    plt.close(fig)

    # Plot 2d trajectories if the environment is 2D.
    # if "Empty" in env_id:
    print(f'Plotting 2D trajectories')
    fig, axs = planner_visualizer.render_robot_trajectories(
        trajs=trajs
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories_2d.png'), dpi=100)
    plt.close(fig)

    # Create animation. Do this with CBS. Yes I know, it's weird.
    # Densify the paths for visualization.
    trajs_l = [traj for traj in trajs]
    trajs_dense_l = densify_trajs(trajs_l, 1)
    # Add batch dimension to all paths.
    trajs_dense_l = [traj.unsqueeze(0) for traj in trajs_dense_l]

    base_file_name = Path(os.path.basename(__file__)).stem
    output_fpath = os.path.join(results_dir, f'{base_file_name}-robot-traj.gif')
    # Render the paths.
    print('Rendering paths and saving to:', output_fpath)
    # Print a link path to terminal that can be clicked to get there.
    file_uri = f'file://{os.path.realpath(output_fpath)}'
    print(f'Click to open GIF: {file_uri}')
    print(f'Click to open output  directory:file://{os.path.realpath(results_dir)}')

    planner_visualizer.animate_multi_robot_trajectories(
        trajs_l=trajs_dense_l,
        start_state_l=[pos_trajs[i][0] for i in range(len(trajs_dense_l))],
        goal_state_l=[pos_trajs[i][-1] for i in range(len(trajs_dense_l))],
        plot_trajs=True,
        video_filepath=output_fpath,
        n_frames=max((2, trajs_dense_l[0].shape[1])),
        # n_frames=pos_trajs_iters[-1].shape[1],
        anim_time=15.0,
        constraints=None,
        colors=[plt.cm.tab10(i) for i in range(len(trajs_dense_l))],
    )

    num_trajectories_coll, num_trajectories_free = len(trajs_last_iter_coll), len(trajs_last_iter_free)
    return num_trajectories_coll, num_trajectories_free


def generate_linear_trajectories(
        env_id,
        robot_id,
        num_trajectories_per_context,
        results_dir,
        threshold_start_goal_pos=1.0,
        obstacle_cutoff_margin=0.03,
        n_tries=1000,
        rrt_max_time=300,
        gpmp_opt_iters=500,
        n_support_points=64,
        duration=5.0,
        tensor_args=None,
        is_wait_at_goal=True,
        debug=False,
):
    # -------------------------------- Load env, robot, task ---------------------------------
    # Environment
    print("linear planner")
    env_class = getattr(environments, env_id)
    env = env_class(tensor_args=tensor_args)

    # Robot
    robot_class = getattr(robots, robot_id)
    robot = robot_class(tensor_args=tensor_args)

    # Task
    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        tensor_args=tensor_args
    )

    # -------------------------------- Start, Goal states ---------------------------------
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    n_trajectories = num_trajectories_per_context

    # -------------------------------- Linear Planner ---------------------------------
    #  The output shape of trajectories is (n_trajectories_for_start_goal_pair, n_support_points, state_dim)
    if is_wait_at_goal:
        # Only allowing velocity v or velocity 0. These are stacked below the positions (vx, vy).
        v_mag = 0.05
    else:
        # Velocity is distance/steps.
        v_mag = torch.linalg.norm(goal_state_pos - start_state_pos) / n_support_points

    if n_trajectories != 1:
        raise ValueError(f"n_trajectories must be 1 for linear planner. Got {n_trajectories}")
    traj_dist = torch.linalg.norm(goal_state_pos - start_state_pos)
    traj_num_points_moving = traj_dist / v_mag
    traj_num_points_moving = torch.floor(traj_num_points_moving).int()
    traj_interpolation_weights = torch.linspace(0, 1, int(traj_num_points_moving), **tensor_args).unsqueeze(1)
    traj = start_state_pos + traj_interpolation_weights * (goal_state_pos - start_state_pos)
    # Add points to the end of the trajectory waiting at the goal, if any exist.
    traj_num_points_waiting = n_support_points - traj_num_points_moving
    if traj_num_points_waiting > 0:
        traj = torch.cat((traj, torch.stack([goal_state_pos] * int(traj_num_points_waiting))))
    # Compute the velocity vectors by finite differencing.
    traj_vel = torch.cat((torch.diff(traj, dim=0), torch.zeros(1, robot.q_dim, **tensor_args)))
    traj = torch.cat((traj.unsqueeze(0), traj_vel.unsqueeze(0)), dim=-1)
    trajs_last_iter = traj

    # -------------------------------- Save trajectories ---------------------------------
    print(f'----------------STATISTICS----------------')
    print(f'percentage free trajs: {task.compute_fraction_free_trajs(trajs_last_iter) * 100:.2f}')
    print(f'percentage collision intensity {task.compute_collision_intensity_trajs(trajs_last_iter) * 100:.2f}')
    print(f'success {task.compute_success_free_trajs(trajs_last_iter)}')

    # save
    torch.cuda.empty_cache()
    trajs_last_iter_coll, trajs_last_iter_free = task.get_trajs_collision_and_free(trajs_last_iter)
    if trajs_last_iter_coll is None:
        trajs_last_iter_coll = torch.empty(0)
    torch.save(trajs_last_iter_coll, os.path.join(results_dir, f'trajs-collision.pt'))
    if trajs_last_iter_free is None:
        trajs_last_iter_free = torch.empty(0)
    torch.save(trajs_last_iter_free, os.path.join(results_dir, f'trajs-free.pt'))

    # Save results data dict.
    results_data_dict = {
        'duration': duration,
        'n_support_points': n_support_points,
        'dt': duration / n_support_points,
        'trajs_iters_coll': None,
        'trajs_iters_free': None,
    }

    with open(os.path.join(results_dir, f'results_data_dict.pickle'), 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------- Visualize ---------------------------------
    planner_visualizer = PlanningVisualizer(task=task)

    trajs = trajs_last_iter_free
    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig, axs = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
    )
    # save figure
    fig.savefig(os.path.join(results_dir, f'trajectories.png'), dpi=100)
    plt.close(fig)

    # Visualize animated. Uncomment below to safe a GIF.
    # trajs_dense_l = densify_trajs([trajs_last_iter], 1)
    # output_fpath = os.path.join(results_dir, f'robot-traj.gif')
    # file_uri = f'file://{os.path.realpath(output_fpath)}'
    # print(f'Click to open output  directory:file://{os.path.realpath(results_dir)}')
    #
    # print(f'Click to open GIF: {file_uri}')
    # planner_visualizer.animate_multi_robot_trajectories(
    #     trajs_l=trajs_dense_l,
    #     start_state_l=[pos_trajs[i][0] for i in range(len(trajs_dense_l))],
    #     goal_state_l=[pos_trajs[i][-1] for i in range(len(trajs_dense_l))],
    #     plot_trajs=True,
    #     video_filepath=output_fpath,
    #     n_frames=max((2, trajs_dense_l[0].shape[1])),
    #     # n_frames=pos_trajs_iters[-1].shape[1],
    #     anim_time=15.0,
    #     constraints=None,
    #     colors=[plt.cm.tab10(i) for i in range(len(trajs_dense_l))],
    # )

    num_trajectories_coll, num_trajectories_free = len(trajs_last_iter_coll), len(trajs_last_iter_free)
    return num_trajectories_coll, num_trajectories_free


@single_experiment_yaml
def experiment(
    # env_id: str = 'EnvEmptyNoWait2D',
    # env_id: str = 'EnvEmpty2D',
    # env_id: str = 'EnvDropRegion2D',
    env_id: str = 'EnvHighways2D',
    # env_id: str = 'EnvConveyor2D',

    robot_id: str = 'RobotPlanarDisk',

    n_support_points: int = 64,
    duration: float = 5.0,  # seconds

    threshold_start_goal_pos: float = 0.5,
    # threshold_start_goal_pos: float = 1.83,

    is_start_goal_near_limits: bool = False,

    obstacle_cutoff_margin: float = 0.05,

    num_trajectories: int = 5,

        # device: str = 'cpu',
    device: str = 'cuda',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = int(time.time()),
    # seed: int = 3,
    # seed: int = 1679258088,
    results_dir: str = f"data",

    #######################################
    **kwargs
):
    if debug:
        fix_random_seed(seed)

    print(f'\n\n-------------------- Generating data --------------------')
    print(f'Seed:  {seed}')
    print(f'Env:   {env_id}')
    print(f'Robot: {robot_id}')
    print(f'num_trajectories: {num_trajectories}')

    ####################################################################################################################
    tensor_args = {'device': device, 'dtype': torch.float32}

    metadata = {
        'env_id': env_id,
        'robot_id': robot_id,
        'num_trajectories': num_trajectories
    }
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)

    num_trajectories_coll, num_trajectories_free = generate_collision_free_trajectories(
        env_id,
        robot_id,
        num_trajectories,
        results_dir,
        threshold_start_goal_pos=threshold_start_goal_pos,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        n_support_points=n_support_points,
        duration=duration,
        is_start_goal_near_limits=is_start_goal_near_limits,
        tensor_args=tensor_args,
        debug=debug
    )

    # Start linear.
    # num_trajectories_coll, num_trajectories_free = generate_linear_trajectories(
    #     env_id,
    #     robot_id,
    #     num_trajectories,
    #     results_dir,
    #     threshold_start_goal_pos=threshold_start_goal_pos,
    #     obstacle_cutoff_margin=obstacle_cutoff_margin,
    #     n_support_points=n_support_points,
    #     duration=duration,
    #     tensor_args=tensor_args,
    #     is_wait_at_goal=False,
    #     debug=debug,
    # )
    # End linear.

    metadata.update(
        num_trajectories_generated=num_trajectories_coll + num_trajectories_free,
        num_trajectories_generated_coll=num_trajectories_coll,
        num_trajectories_generated_free=num_trajectories_free,
    )
    with open(os.path.join(results_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, Dumper=yaml.Dumper)


if __name__ == '__main__':
    run_experiment(experiment)
