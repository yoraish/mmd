"""
Modified from https://github.com/jacarvalho/mpd-public
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

import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraint, CostMaxVelocity
from mmd.models import TemporalUnet, UNET_DIM_MULTS
from mmd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mmd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mmd.trainer import get_dataset, get_model
from mmd.utils.loading import load_params_from_yaml
from torch_robotics.robots import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from mmd.planners.single_agent.common import PlannerOutput
from mmd.planners.single_agent.single_agent_planner_base import SingleAgentPlanner
from mmd.common.experiences import PathExperience, PathBatchExperience
from mmd.common.constraints import MultiPointConstraint
from mmd.common.pretty_print import *


class MPD(SingleAgentPlanner):
    """
    A class that allows repeated calls to the same model with different inputs.
    This class keeps track of constraints and feeds them to the model only when needed.
    """

    def __init__(self,
                 model_id: str,
                 planner_alg: str,
                 start_state_pos: torch.tensor,
                 goal_state_pos: torch.tensor,
                 use_guide_on_extra_objects_only: bool,
                 start_guide_steps_fraction: float,
                 n_guide_steps: int,
                 n_diffusion_steps_without_noise: int,
                 weight_grad_cost_collision: float,
                 weight_grad_cost_smoothness: float,
                 weight_grad_cost_constraints: float,
                 weight_grad_cost_soft_constraints: float,
                 factor_num_interpolated_points_for_collision: float,
                 trajectory_duration: float,
                 device: str,
                 debug: bool,
                 seed: int,
                 results_dir: str,
                 trained_models_dir: str,
                 n_samples: int,
                 n_local_inference_noising_steps: int,
                 n_local_inference_denoising_steps: int,
                 **kwargs
                 ):
        super().__init__()
        # The constraints are stored here. This is a list of ConstraintCost.
        self.constraints = []
        self.weight_grad_cost_constraints = weight_grad_cost_constraints
        self.weight_grad_cost_soft_constraints = weight_grad_cost_soft_constraints

        ####################################
        fix_random_seed(seed)

        device = get_torch_device(device)
        tensor_args = {'device': device, 'dtype': torch.float32}
        ####################################
        print(f'####################################')
        print(f'Initializing Planner with Model -- {model_id}')
        print(f'Algorithm -- {planner_alg}')
        run_prior_only = False
        run_prior_then_guidance = False
        if planner_alg == 'mmd':
            pass
        elif planner_alg == 'diffusion_prior_then_guide':
            run_prior_then_guidance = True
        elif planner_alg == 'diffusion_prior':
            run_prior_only = True
        else:
            raise NotImplementedError

        ####################################
        model_dir = os.path.join(trained_models_dir, model_id)
        results_dir = os.path.join(model_dir, 'results_inference', str(seed))
        os.makedirs(results_dir, exist_ok=True)

        args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

        ####################################
        # Load dataset with env, robot, task. The TrajectoryDataset type is used here.
        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=True,
            obstacle_cutoff_margin=0.05,
            **args,
            tensor_args=tensor_args
        )
        # Extract objects from the dataset.
        dataset = train_subset.dataset
        # Number of support points in the trajectory.
        n_support_points = dataset.n_support_points
        # The environment.
        env = dataset.env
        # The robot. Contains the dt, the joint limits, etc.
        robot = dataset.robot
        # The task, commonly PlanningTask, is in charge of objects, extra objects, collisions, etc.
        task = dataset.task

        dt = trajectory_duration / n_support_points  # time interval for finite differences

        # set robot's dt
        robot.dt = dt

        ####################################
        # Load prior model
        diffusion_configs = dict(
            variance_schedule=args['variance_schedule'],
            n_diffusion_steps=args['n_diffusion_steps'],
            predict_epsilon=args['predict_epsilon'],
        )
        unet_configs = dict(
            state_dim=dataset.state_dim,
            n_support_points=dataset.n_support_points,
            unet_input_dim=args['unet_input_dim'],
            dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
        )
        diffusion_model = get_model(
            model_class=args['diffusion_model_class'],
            model=TemporalUnet(**unet_configs),
            tensor_args=tensor_args,
            **diffusion_configs,
            **unet_configs
        )
        diffusion_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args[
                'use_ema'] else 'model_current_state_dict.pth'),
                       map_location=tensor_args['device'])
        )
        diffusion_model.eval()
        model = diffusion_model

        freeze_torch_model_params(model)
        model = torch.compile(model)
        model.warmup(horizon=n_support_points, device=device)

        ####################################
        # If the args specify a test start and goal, use those.
        if start_state_pos is not None and goal_state_pos is not None:
            print(f'start_state_pos: {start_state_pos}')
            print(f'goal_state_pos: {goal_state_pos}')
        else:
            # Random initial and final positions
            n_tries = 100
            start_state_pos, goal_state_pos = None, None
            for _ in range(n_tries):
                q_free = task.random_coll_free_q(n_samples=2)
                start_state_pos = q_free[0]
                goal_state_pos = q_free[1]

                if torch.linalg.norm(start_state_pos - goal_state_pos) > dataset.threshold_start_goal_pos:
                    break

        if start_state_pos is None or goal_state_pos is None:
            raise ValueError(f"No collision free configuration was found\n"
                             f"start_state_pos: {start_state_pos}\n"
                             f"goal_state_pos:  {goal_state_pos}\n")

        print(f'start_state_pos: {start_state_pos}')
        print(f'goal_state_pos: {goal_state_pos}')

        ####################################
        # Run motion planning inference

        ########
        # normalize start and goal positions
        hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
        context = None

        ########
        # Set up the planning costs
        # Cost collisions
        cost_collision_l = []
        weights_grad_cost_l = []  # for guidance, the weights_cost_l are the gradient multipliers (after gradient clipping)
        if use_guide_on_extra_objects_only:
            collision_fields = task.get_collision_fields_extra_objects()
        else:
            collision_fields = task.get_collision_fields()

        for collision_field in collision_fields:
            cost_collision_l.append(
                CostCollision(
                    robot, n_support_points,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=tensor_args
                )
            )
            weights_grad_cost_l.append(weight_grad_cost_collision)

        # Cost smoothness
        cost_smoothness_l = [
            CostGPTrajectory(
                robot, n_support_points, dt, sigma_gp=1.0,
                tensor_args=tensor_args
            )
        ]
        weights_grad_cost_l.append(weight_grad_cost_smoothness)

        ####### Cost composition
        cost_func_list = [
            *cost_collision_l,
            *cost_smoothness_l,
            # *cost_max_velocity_l,
        ]
        # A `*cost_constraints_l` will be added as "extra cost" and removed after each planning call.

        cost_composite = CostComposite(
            robot, n_support_points, cost_func_list,
            weights_cost_l=weights_grad_cost_l,
            tensor_args=tensor_args
        )

        ########
        # Guiding manager
        guide = GuideManagerTrajectoriesWithVelocity(
            dataset,
            cost_composite,
            clip_grad=True,
            interpolate_trajectories_for_collision=True,
            num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision),
            tensor_args=tensor_args,
        )

        t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)

        # Keep some variables in the class as members.
        self.start_state_pos = torch.clone(start_state_pos)
        self.goal_state_pos = torch.clone(goal_state_pos)
        self.robot = robot
        self.context = context
        self.run_prior_only = run_prior_only
        self.run_prior_then_guidance = run_prior_then_guidance
        self.n_diffusion_steps_without_noise = n_diffusion_steps_without_noise
        self.hard_conds = hard_conds
        self.model = model
        self.n_support_points = n_support_points
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.guide = guide
        self.tensor_args = tensor_args
        # Batch-size. How many trajectories to generate at once.
        self.num_samples = n_samples
        # When doing local inference, how many steps to add noise for before denoising again.
        self.n_local_inference_noising_steps = n_local_inference_noising_steps  # n_local_inference_noising_steps
        self.n_local_inference_denoising_steps = n_local_inference_denoising_steps
        # Dataset.
        self.dataset = dataset
        # Task, e.g., planning task.
        self.task = task
        # Directories.
        self.results_dir = results_dir

        # Cache of previous call data.
        self.recent_call_data = PlannerOutput()

        self.sample_fn_kwargs = dict(
                 guide=None if self.run_prior_then_guidance or self.run_prior_only else self.guide,
                 n_guide_steps=self.n_guide_steps,
                 t_start_guide=self.t_start_guide,
                 noise_std_extra_schedule_fn=lambda x: 0.5,
        )

    def __call__(self, start_state_pos, goal_state_pos, constraints_l=None, experience: PathBatchExperience = None,
                 *args,
                 **kwargs):
        """
        Call the model with the given parameters.
        :param n_samples: Number of trajectories to generate.
        :param start_state_pos: The start state of the robot.
        :param goal_state_pos: The goal state of the robot.
        :param constraints_l: A list of constraints.
        :param previous_path: The previous path of the robot. This would be used to guide the next path.
        """
        # Check that the requested start and goal states are similar to the ones stored.
        if not torch.allclose(start_state_pos, self.start_state_pos):
            raise ValueError("The start state is different from the one stored in the planner.")
        if not torch.allclose(goal_state_pos, self.goal_state_pos):
            raise ValueError("The goal state is different from the one stored in the planner.")

        # Process the constraints into cost components.
        if constraints_l is not None:
            print("Planning with " + str(len(constraints_l)) + " constraints.")
        else:
            print("Planning without constraints.")

        cost_constraints_l = []
        if constraints_l is not None:
            for c in constraints_l:
                cost_constraints_l.append(
                    CostConstraint(
                        self.robot,
                        self.n_support_points,
                        q_l=c.get_q_l(),
                        traj_range_l=c.get_t_range_l(),
                        radius_l=c.radius_l,
                        is_soft=c.is_soft,
                        tensor_args=self.tensor_args
                    )
                )
        # Carry out inference with the constraints. If there is no experience, inference from scratch.
        with TimerCUDA() as timer_inference:
            if experience is None:
                trajs_normalized_iters, _, _ = self.run_constrained_inference(
                    cost_constraints_l)  # Shape [B (n_samples), H, D]
            # Otherwise, use the experience path as a seed for a local inference call.
            else:
                trajs_normalized_iters, _, _ = self.run_constrained_local_inference(cost_constraints_l, experience)
        t_total = timer_inference.elapsed

        # Un-normalize trajectory samples from the models.
        trajs_iters = self.dataset.unnormalize_trajectories(trajs_normalized_iters)

        trajs_final = trajs_iters[-1]
        trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = (
            self.task.get_trajs_collision_and_free(trajs_final, return_indices=True))

        # Compute best traj.
        idx_best_traj = None  # Index of the best trajectory in the list of all trajectories (trajs_final).
        idx_best_free_traj = None  # Index of the best trajectory in the list of all free trajs (trajs_final_free).
        cost_best_free_traj = None  # Cost of the best trajectory.
        cost_smoothness = None  # Cost of smoothness for all free trajectories.
        cost_path_length = None  # Cost of path length for all free trajectories.
        cost_all = None  # Cost of all factors for all free trajectories. This is a combination of some costs.
        variance_waypoint_trajs_final_free = None  # Variance of waypoints for all free trajectories.
        if trajs_final_free is not None:
            cost_smoothness = compute_smoothness(trajs_final_free, self.robot)
            print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

            cost_path_length = compute_path_length(trajs_final_free, self.robot)
            print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

            # Compute best trajectory.
            cost_all = cost_path_length + cost_smoothness
            idx_best_free_traj = torch.argmin(cost_all).item()
            idx_best_traj = trajs_final_free_idxs[idx_best_free_traj]
            cost_best_free_traj = torch.min(cost_all).item()
            print(f'cost best: {cost_best_free_traj:.3f}')

            variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, self.robot)

        self.recent_call_data = PlannerOutput()
        self.recent_call_data.trajs_iters = trajs_iters
        self.recent_call_data.trajs_final = trajs_final
        self.recent_call_data.trajs_final_coll = trajs_final_coll
        self.recent_call_data.trajs_final_coll_idxs = trajs_final_coll_idxs
        self.recent_call_data.trajs_final_free = trajs_final_free  # Shape [B, H, D]
        self.recent_call_data.t_total = t_total
        self.recent_call_data.trajs_final_free_idxs = trajs_final_free_idxs  # Shape [B]
        self.recent_call_data.idx_best_traj = idx_best_traj
        self.recent_call_data.idx_best_free_traj = idx_best_free_traj
        self.recent_call_data.cost_best_free_traj = cost_best_free_traj
        self.recent_call_data.cost_smoothness = cost_smoothness
        self.recent_call_data.cost_path_length = cost_path_length
        self.recent_call_data.cost_all = cost_all
        self.recent_call_data.variance_waypoint_trajs_final_free = variance_waypoint_trajs_final_free
        self.recent_call_data.constraints_l = constraints_l

        # Smooth the trajectories in trajs_final.
        if self.recent_call_data.trajs_final is not None:
            self.recent_call_data.trajs_final = smooth_trajs(self.recent_call_data.trajs_final)

        return self.recent_call_data

    def run_constrained_inference(self, cost_constraints_l: List[CostConstraint]):
        # Add these cost factors, alongside their weights as specified, to the `guide` object of the model.
        self.guide.add_extra_costs(cost_constraints_l,
                                   [self.weight_grad_cost_soft_constraints if c.is_soft else
                                    self.weight_grad_cost_constraints
                                    for c in cost_constraints_l])

        # Sample trajectories with the diffusion/cvae model
        with TimerCUDA() as timer_model_sampling:
            trajs_normalized_iters = self.model.run_inference(
                self.context, self.hard_conds,
                n_samples=self.num_samples, horizon=self.n_support_points,
                return_chain=True,
                sample_fn=ddpm_sample_fn,
                **self.sample_fn_kwargs,
                n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
                # ddim=True
            )
        t_model_sampling = timer_model_sampling.elapsed
        print(f't_model_sampling: {t_model_sampling:.3f} sec')

        ########
        # run extra guiding steps without diffusion
        t_post_diffusion_guide = 0.0
        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide +
                                            self.n_diffusion_steps_without_noise) * self.n_guide_steps
            print(CYAN + f'Running extra guiding steps without diffusion. Num steps:', n_post_diffusion_guide_steps,
                  RESET)
            with TimerCUDA() as timer_post_model_sample_guide:
                trajs = trajs_normalized_iters[-1]
                trajs_post_diff_l = []
                for i in range(n_post_diffusion_guide_steps):
                    trajs = guide_gradient_steps(
                        trajs,
                        hard_conds=self.hard_conds,
                        guide=self.guide,
                        n_guide_steps=1,
                        unnormalize_data=False
                    )
                    trajs_post_diff_l.append(trajs)

                chain = torch.stack(trajs_post_diff_l, dim=1)
                chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
                trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
            t_post_diffusion_guide = timer_post_model_sample_guide.elapsed
            print(f't_post_diffusion_guide: {t_post_diffusion_guide:.3f} sec')

        # Remove the extra cost.
        self.guide.reset_extra_costs()

        return trajs_normalized_iters, t_model_sampling, t_post_diffusion_guide

    def run_constrained_local_inference(self, cost_constraints_l: List[CostConstraint],
                                        experience: PathBatchExperience):
        # Add these cost factors, alongside their weights as specified, to the `guide` object of the model.
        self.guide.add_extra_costs(cost_constraints_l,
                                   [self.weight_grad_cost_soft_constraints if c.is_soft else
                                    self.weight_grad_cost_constraints
                                    for c in cost_constraints_l])

        # Sample trajectories with the diffusion/cvae model
        with TimerCUDA() as timer_model_sampling:
            trajs_normalized_iters = self.model.run_local_inference(
                experience.path_b,
                self.n_local_inference_noising_steps,
                self.n_local_inference_denoising_steps,
                self.context,
                self.hard_conds,
                n_samples=self.num_samples,
                horizon=self.n_support_points,
                return_chain=True,
                sample_fn=ddpm_sample_fn,
                **self.sample_fn_kwargs,
                n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
                # ddim=True
            )
        t_model_sampling = timer_model_sampling.elapsed
        print(f't_model_sampling: {t_model_sampling:.3f} sec')

        ########
        # Run extra guiding steps without diffusion. This is only done if variable `run_prior_then_guidance` is True.
        t_post_diffusion_guide = 0.0
        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide +
                                            self.n_diffusion_steps_without_noise) * self.n_guide_steps
            print(CYAN + f'Running extra guiding steps without diffusion. Num steps:', n_post_diffusion_guide_steps,
                  RESET)
            with TimerCUDA() as timer_post_model_sample_guide:
                trajs = trajs_normalized_iters[-1]
                trajs_post_diff_l = []
                for i in range(n_post_diffusion_guide_steps):
                    trajs = guide_gradient_steps(
                        trajs,
                        hard_conds=self.hard_conds,
                        guide=self.guide,
                        n_guide_steps=1,
                        unnormalize_data=False
                    )
                    trajs_post_diff_l.append(trajs)

                chain = torch.stack(trajs_post_diff_l, dim=1)
                chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
                trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
            t_post_diffusion_guide = timer_post_model_sample_guide.elapsed
            print(f't_post_diffusion_guide: {t_post_diffusion_guide:.3f} sec')

        # Remove the extra cost.
        self.guide.reset_extra_costs()

        return trajs_normalized_iters, t_model_sampling, t_post_diffusion_guide

    def save_recent_result(self):
        # Compute motion planning metrics
        print(f'\n----------------METRICS----------------')
        print(f't_total: {self.recent_call_data.t_total:.3f} sec')

        success_free_trajs = self.task.compute_success_free_trajs(self.recent_call_data.trajs_final)
        fraction_free_trajs = self.task.compute_fraction_free_trajs(self.recent_call_data.trajs_final)
        collision_intensity_trajs = self.task.compute_collision_intensity_trajs(self.recent_call_data.trajs_final)

        print(f'success: {success_free_trajs}')
        print(f'percentage free trajs: {fraction_free_trajs * 100:.2f}')
        print(f'percentage collision intensity: {collision_intensity_trajs * 100:.2f}')

        # compute costs only on collision-free trajectories
        traj_final_free_best = None
        idx_best_traj = None
        cost_best_free_traj = None
        cost_smoothness = None
        cost_path_length = None
        cost_all = None
        variance_waypoint_trajs_final_free = None
        if self.recent_call_data.trajs_final_free is not None:
            print(f'cost smoothness: {self.recent_call_data.cost_smoothness.mean():.4f}, '
                  f'{self.recent_call_data.cost_smoothness.std():.4f}')
            print(f'cost path length: {self.recent_call_data.cost_path_length.mean():.4f}, '
                  f'{self.recent_call_data.cost_path_length.std():.4f}')

            # compute best trajectory
            cost_best_free_traj = torch.min(self.recent_call_data.cost_all).item()
            print(f'cost best: {cost_best_free_traj:.3f}')

            # variance of waypoints
            print(f'variance waypoint: {self.recent_call_data.variance_waypoint_trajs_final_free:.4f}')

        print(f'\n--------------------------------------\n')

        ####################################
        # Save data
        results_data_dict = {
            'trajs_iters': self.recent_call_data.trajs_iters,
            'trajs_final_coll': self.recent_call_data.trajs_final_coll,
            'trajs_final_coll_idxs': self.recent_call_data.trajs_final_coll_idxs,
            'trajs_final_free': self.recent_call_data.trajs_final_free,
            'trajs_final_free_idxs': self.recent_call_data.trajs_final_free_idxs,
            'success_free_trajs': success_free_trajs,
            'fraction_free_trajs': fraction_free_trajs,
            'collision_intensity_trajs': collision_intensity_trajs,
            'idx_best_traj': idx_best_traj,
            'traj_final_free_best': traj_final_free_best,
            'cost_best_free_traj': cost_best_free_traj,
            'cost_path_length_trajs_final_free': cost_path_length,
            'cost_smoothness_trajs_final_free': cost_smoothness,
            'cost_all_trajs_final_free': cost_all,
            'variance_waypoint_trajs_final_free': variance_waypoint_trajs_final_free,
            't_total': self.recent_call_data.t_total
        }
        with open(os.path.join(self.results_dir, 'results_data_dict.pickle'), 'wb') as handle:
            pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def render_recent_result(self, animation_duration: float = 5.0):
        # Render
        planner_visualizer = PlanningVisualizer(
            task=self.task,
        )

        base_file_name = Path(os.path.basename(__file__)).stem
        traj_final_free_best = self.recent_call_data.trajs_final[self.recent_call_data.idx_best_traj]
        pos_trajs_iters = self.robot.get_position(self.recent_call_data.trajs_iters)

        planner_visualizer.animate_opt_iters_joint_space_state(
            trajs=self.recent_call_data.trajs_iters,
            pos_start_state=self.start_state_pos, pos_goal_state=self.goal_state_pos,
            vel_start_state=torch.zeros_like(self.start_state_pos),
            vel_goal_state=torch.zeros_like(self.goal_state_pos),
            traj_best=traj_final_free_best,
            video_filepath=os.path.join(self.results_dir, f'{base_file_name}-joint-space-opt-iters.gif'),
            n_frames=max((2, len(self.recent_call_data.trajs_iters) // 10)),
            anim_time=5
        )

        # visualize in the planning environment
        planner_visualizer.animate_opt_iters_robots(
            trajs=pos_trajs_iters, start_state=self.start_state_pos, goal_state=self.goal_state_pos,
            traj_best=traj_final_free_best,
            video_filepath=os.path.join(self.results_dir, f'{base_file_name}-traj-opt-iters.gif'),
            # n_frames=max((2, len(trajs_iters))),
            n_frames=2,
            anim_time=5
        )

        planner_visualizer.animate_robot_trajectories(
            trajs=pos_trajs_iters[-1], start_state=self.start_state_pos, goal_state=self.goal_state_pos,
            plot_trajs=True,
            video_filepath=os.path.join(self.results_dir, f'{base_file_name}-robot-traj.gif'),
            n_frames=max((2, pos_trajs_iters[-1].shape[1])),
            # n_frames=pos_trajs_iters[-1].shape[1],
            anim_time=animation_duration,
            constraints=self.recent_call_data.constraints_l
        )
