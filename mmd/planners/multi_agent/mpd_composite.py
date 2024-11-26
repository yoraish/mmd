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
import os
from math import ceil
from pathlib import Path
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
# Project imports.
from experiment_launcher.utils import fix_random_seed
from mmd.common.experiments import TrialSuccessStatus
from mmd.common.constraints import MultiPointConstraint
from mmd.common.multi_agent_utils import *
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
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
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes

allow_ops_in_compiled_graph()

TRAINED_MODELS_DIR = '../../data_trained_models/'


class MPDComposite:
    def __init__(self,
                 low_level_planner_l: List,  # Not used.
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 model_id: str,
                 reference_robot=None,
                 reference_task=None,
                 planner_alg: str = 'mmd',
                 use_guide_on_extra_objects_only: bool = False,
                 n_samples: int = 64,
                 start_guide_steps_fraction: float = 0.5,
                 n_guide_steps: int = 25,
                 n_diffusion_steps_without_noise: int = 5,
                 weight_grad_cost_collision: float = 5e-2,
                 weight_grad_cost_smoothness: float = 1e-2,
                 factor_num_interpolated_points_for_collision: float = 1.5,
                 trajectory_duration: float = 5.0,
                 device: str = 'cuda',
                 seed: int = 18,
                 **kwargs):

        # Some parameters:
        self.num_agents = len(start_l)
        self.agent_color_l = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))
        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        self.model_id = model_id

        # Keep a reference robot for collision checking in a group of robots.
        self.reference_robot = reference_robot
        self.reference_task = reference_task
        self.tensor_args = params.tensor_args
        # Reshape the start and goal states (list of q_dim tensors) to a (n_agents, q_dim) tensor.
        self.start_state_pos = torch.cat(self.start_state_pos_l, dim=0)
        self.goal_state_pos = torch.cat(self.goal_state_pos_l, dim=0)
        # Results dir.
        self.results_dir = kwargs['results_dir']

        # Prepare for planning query here in the constructor.
        fix_random_seed(seed)
        self.n_diffusion_steps_without_noise = n_diffusion_steps_without_noise
        self.n_guide_steps = n_guide_steps
        self.n_samples = n_samples

        device = get_torch_device(device)
        tensor_args = {'device': device, 'dtype': torch.float32}

        print(
            f'####################################')
        print(f'Model -- {self.model_id}')
        print(f'Algorithm -- {planner_alg}')
        run_prior_only = False
        self.run_prior_then_guidance = False
        if planner_alg == 'mmd':
            pass
        elif planner_alg == 'diffusion_prior_then_guide':
            self.run_prior_then_guidance = True
        elif planner_alg == 'diffusion_prior':
            run_prior_only = True
        else:
            raise NotImplementedError

        ####################################
        model_dir = os.path.join(TRAINED_MODELS_DIR, self.model_id)
        results_dir = os.path.join(model_dir, 'results_inference', str(seed))
        os.makedirs(results_dir, exist_ok=True)

        args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

        ####################################
        # Load dataset with env, robot, and task.
        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=True,
            obstacle_cutoff_margin=0.01,
            # Important for having the self-collision cost affect robots that are not directly overlapping too.
            **args,
            tensor_args=tensor_args
        )
        self.dataset = train_subset.dataset
        self.n_support_points = self.dataset.n_support_points
        env = self.dataset.env
        self.robot = self.dataset.robot
        self.task = self.dataset.task

        dt = trajectory_duration / self.n_support_points

        # set robot's dt
        self.robot.dt = dt

        self.reference_task = self.task
        self.reference_robot = self.robot

        ####################################
        # Load prior model
        diffusion_configs = dict(
            variance_schedule=args['variance_schedule'],
            n_diffusion_steps=args['n_diffusion_steps'],
            predict_epsilon=args['predict_epsilon'],
        )
        unet_configs = dict(
            state_dim=self.dataset.state_dim,
            n_support_points=self.dataset.n_support_points,
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
        self.model = diffusion_model

        freeze_torch_model_params(self.model)
        self.model = torch.compile(self.model)
        self.model.warmup(horizon=self.n_support_points, device=device)

        ####################################
        print(f'start_state_pos: {self.start_state_pos}')
        print(f'goal_state_pos:  {self.goal_state_pos}')
        ####################################
        # Run motion planning inference.

        ########
        # Normalize start and goal positions.
        self.hard_conds = self.dataset.get_hard_conditions(torch.vstack((self.start_state_pos, self.goal_state_pos)), normalize=True)
        self.context = None

        ########
        # Set up the planning costs.
        # Cost collisions.
        cost_collision_l = []
        weights_grad_cost_l = []
        if use_guide_on_extra_objects_only:
            collision_fields = self.task.get_collision_fields_extra_objects()
        else:
            collision_fields = self.task.get_collision_fields()

        for collision_field in collision_fields:
            cost_collision_l.append(
                CostCollision(
                    self.robot, self.n_support_points,
                    field=collision_field,
                    sigma_coll=1.0,
                    tensor_args=tensor_args
                )
            )
            weights_grad_cost_l.append(weight_grad_cost_collision)

        # Cost smoothness.
        cost_smoothness_l = [
            CostGPTrajectory(
                self.robot, self.n_support_points, dt, sigma_gp=1.0,
                tensor_args=tensor_args
            )
        ]
        weights_grad_cost_l.append(weight_grad_cost_smoothness)

        ####### Cost composition.
        cost_func_list = [
            *cost_collision_l,
            *cost_smoothness_l
        ]

        cost_composite = CostComposite(
            self.robot, self.n_support_points, cost_func_list,
            weights_cost_l=weights_grad_cost_l,
            tensor_args=tensor_args
        )

        ########
        # Guiding manager.
        self.guide = GuideManagerTrajectoriesWithVelocity(
            self.dataset,
            cost_composite,
            clip_grad=True,
            interpolate_trajectories_for_collision=True,
            num_interpolated_points=ceil(self.n_support_points * factor_num_interpolated_points_for_collision),
            tensor_args=tensor_args,
        )

        self.t_start_guide = ceil(start_guide_steps_fraction * self.model.n_diffusion_steps)
        self.sample_fn_kwargs = dict(
            guide=None if self.run_prior_then_guidance or run_prior_only else self.guide,
            n_guide_steps=self.n_guide_steps,
            t_start_guide=self.t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )

    def render_paths(self, paths_l: List[torch.Tensor], constraints_l: List[MultiPointConstraint] = None,
                     plot_trajs=False,
                     animation_duration: float = 10.0, output_fpath=None, n_frames=None, show_robot_in_image=True):
        planner_visualizer = PlanningVisualizer(
            task=self.reference_task,
        )

        # Get rid of velocities. This is a hack for now.
        paths_l = [path[:, :2] for path in paths_l]
        trajs = torch.cat(paths_l, dim=1).unsqueeze(0)

        # If animation duration is 0 then only make a picture.
        if animation_duration == 0:
            fig, ax = create_fig_and_axes()
            planner_visualizer.render_robot_trajectories(
                fig=fig,
                ax=ax,
                trajs=trajs,
                start_state=self.start_state_pos,
                goal_state=self.goal_state_pos,
                colors=self.agent_color_l,
                show_robot_in_image=show_robot_in_image
            )
            if output_fpath is None:
                output_fpath = os.path.join(self.results_dir, 'robot-traj.png')
            if output_fpath[-4:] != '.png':
                output_fpath += '.png'
            print(f'Saving image to: file://{output_fpath}')
            plt.axis('off')
            plt.savefig(output_fpath, dpi=300, bbox_inches='tight', pad_inches=0)
            return

        base_file_name = Path(os.path.basename(__file__)).stem

        planner_visualizer.animate_robot_trajectories(
            trajs=trajs,
            start_state=self.start_state_pos,
            goal_state=self.goal_state_pos,
            plot_trajs=plot_trajs,
            video_filepath=os.path.join(self.results_dir, f'{base_file_name}-robot-traj.gif'),
            # n_frames=max((2, trajs_final_free.shape[1]//10)),
            n_frames=trajs.shape[1],
            anim_time=animation_duration
        )
        print(f"file://{os.path.abspath(self.results_dir)}/{base_file_name}-robot-traj.gif")

    def plan(self,
             runtime_limit=1000,
             **kwargs
             ):

        ########
        # Sample trajectories with the diffusion/cvae model
        with TimerCUDA() as timer_model_sampling:
            trajs_normalized_iters = self.model.run_inference(
                self.context, self.hard_conds,
                n_samples=self.n_samples, horizon=self.n_support_points,
                return_chain=True,
                sample_fn=ddpm_sample_fn,
                **self.sample_fn_kwargs,
                n_diffusion_steps_without_noise=self.n_diffusion_steps_without_noise,
                # ddim=True
            )
        print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
        t_total = timer_model_sampling.elapsed

        ########
        # run extra guiding steps without diffusion
        if self.run_prior_then_guidance:
            n_post_diffusion_guide_steps = (self.t_start_guide + self.n_diffusion_steps_without_noise) * self.n_guide_steps
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
            print(f't_post_diffusion_guide: {timer_post_model_sample_guide.elapsed:.3f} sec')
            t_total = timer_model_sampling.elapsed + timer_post_model_sample_guide.elapsed

        # Unnormalize trajectory samples from the models.
        trajs_iters = self.dataset.unnormalize_trajectories(trajs_normalized_iters)

        trajs_final = trajs_iters[-1]
        trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = self.task.get_trajs_collision_and_free(
            trajs_final, return_indices=True)

        ####################################
        # Compute motion planning metrics
        print(f'\n----------------METRICS----------------')
        print(f't_total: {t_total:.3f} sec')

        success_free_trajs = self.task.compute_success_free_trajs(trajs_final)
        fraction_free_trajs = self.task.compute_fraction_free_trajs(trajs_final)
        collision_intensity_trajs = self.task.compute_collision_intensity_trajs(trajs_final)

        print(f'success: {success_free_trajs}')
        print(f'percentage free trajs: {fraction_free_trajs * 100:.2f}')
        print(f'percentage collision intensity: {collision_intensity_trajs * 100:.2f}')
        # Choose the best trajectory among the free trajectories.
        if trajs_final_free is not None:
            cost_smoothness = compute_smoothness(trajs_final_free, self.robot)
            print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

            cost_path_length = compute_path_length(trajs_final_free, self.robot)
            print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

            # compute best trajectory
            cost_all_free = cost_path_length + cost_smoothness
            idx_best_traj_free = torch.argmin(cost_all_free).item()
            traj_final_free_best = trajs_final_free[idx_best_traj_free]
            cost_best_free_traj = torch.min(cost_all_free).item()
            print(f'cost best: {cost_best_free_traj:.3f}')

            # variance of waypoints
            variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, self.robot)
            print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')
        else:
            # TODO compute num collisions.
            return [], 0, TrialSuccessStatus.FAIL_NO_SOLUTION, 0

        print(f'\n--------------------------------------\n')
        ####################################
        # Return the best_path_l, num_ct_expansions, success_status, len(state.conflict_l)
        best_path = trajs_final_free[idx_best_traj_free:idx_best_traj_free + 1].squeeze(0)  # Shape (H, n_agents * q_posvel_dim)
        # Convert to list of tensors, each of shape (n_agents, q_dim). Start by viewing as (n_agents, H, q_dim).
        path_l = []
        for agent_id in range(self.num_agents):
            path = torch.zeros((best_path.shape[0], 2*2), **self.tensor_args)
            path[:, :2] = best_path[:, agent_id * 2:(agent_id + 1) * 2]
            path[:, 2:] = best_path[:,
                                    (self.num_agents + agent_id) * 2:(self.num_agents + agent_id + 1) * 2]
            path_l.append(path)

        # Smooth the path.
        path_l = smooth_trajs(path_l)

        # Check if the path has collisions.
        num_collisions_in_solution = 0
        for t in range(len(path_l[0])):
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    if torch.norm(path_l[i][t, :2] - path_l[j][t, :2]) < 2 * params.robot_planar_disk_radius:
                        # This should be reference_robot.radius.
                        num_collisions_in_solution += 1
        if num_collisions_in_solution > 0:
            print(f'num_collisions_in_solution: {num_collisions_in_solution}\n\n\n\n\n\n\n')
            return path_l, 0, TrialSuccessStatus.FAIL_COLLISION_AGENTS, num_collisions_in_solution

        return path_l, 0, TrialSuccessStatus.SUCCESS, num_collisions_in_solution
