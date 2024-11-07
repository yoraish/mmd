import einops
import torch
from typing import List

from mp_baselines.planners.base import MPPlanner
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import tensor_linspace_v1
from torch_robotics.trajectory.utils import smoothen_trajectory, finite_difference_vector, smoothen_trajectory_with_bc

class HybridPlanner(MPPlanner):
    """
    Runs a sampled-based planner to get an initial trajectory in position, followed by an optimization-based planner
    to optimize for positions and velocities.
    """

    def __init__(
            self,
            sample_based_planners: List,
            opt_based_planner,
            **kwargs
    ):
        super().__init__(
            "HybridSampleAndOptimizationPlanner",
            **kwargs
        )

        # A list of sampling based planners. Those already encode the start and goal states.
        # Each planner is between different start and goal states.
        # If there are more than one planners there, then the paths are stacked (path1, path2, ..., pathN).
        self.sample_based_planners = sample_based_planners
        self.opt_based_planner = opt_based_planner

    def render(self, ax, **kwargs):
        raise NotImplementedError

    def optimize(self, debug=False, print_times=False, return_iterations=False, **kwargs):
        with TimerCUDA() as t_hybrid:
            #################################################
            # Get initial position solutions with a sample-based planner
            # Optimize in parallel
            with TimerCUDA() as t_sample_based:
                if len(self.sample_based_planners) > 1:
                    trajs_l = []
                    for planner in self.sample_based_planners:
                        traj_l_planner = planner.optimize(refill_samples_buffer=True, debug=debug, **kwargs)

                        # Replace any None entry with a linear interpolation.
                        traj_l = []
                        for traj in traj_l_planner:
                            if traj is not None:
                                traj_l.append(traj)
                            else:
                                # If no solution was found, create a linear interpolated trajectory between start and
                                # finish, even if is not collision-free
                                traj = tensor_linspace_v1(
                                    planner.start_state_pos, planner.goal_state_pos, steps=10).T
                                traj_l.append(traj)
                        trajs_l.append(traj_l)

                    traj_l = [torch.cat([trajs_l[p][t] for p in range(len(trajs_l))]) for t in range(len(trajs_l[0]))]

                else:
                    traj_l = self.sample_based_planners[0].optimize(refill_samples_buffer=True, debug=debug, **kwargs)

            if debug or print_times:
                print(f'Sample-based Planner -- Optimization time: {t_sample_based.elapsed:.3f} sec')

            #################################################
            # Interpolate initial trajectory to desired trajectory length, smooth and set average velocity
            traj_pos_vel_l = []
            for traj in traj_l:
                # If no solution was found, create a linear interpolated trajectory between start and finish, even
                # if is not collision-free
                if traj is None:
                    traj = tensor_linspace_v1(
                        self.sample_based_planners[0].start_state_pos, self.sample_based_planners[-1].goal_state_pos,
                        steps=self.opt_based_planner.n_support_points).T
                if 'bc_grad_start' in kwargs and 'bc_grad_goal' in kwargs:
                    print("=====\nUsing bc_grad_start and bc_grad_goal")
                    traj_pos, traj_vel = smoothen_trajectory_with_bc(
                        traj, grad_start=kwargs['bc_grad_start'], grad_goal=kwargs['bc_grad_goal'],
                        n_support_points=self.opt_based_planner.n_support_points, dt=self.opt_based_planner.dt,
                        set_average_velocity=True, zero_velocity=False, tensor_args=self.tensor_args
                    )
                else:
                    traj_pos, traj_vel = smoothen_trajectory(
                        traj, n_support_points=self.opt_based_planner.n_support_points, dt=self.opt_based_planner.dt,
                        set_average_velocity=True, zero_velocity=False, tensor_args=self.tensor_args
                    )

                # Reshape for gpmp/sgpmp interface
                initial_traj_pos_vel = torch.cat((traj_pos, traj_vel), dim=-1)

                traj_pos_vel_l.append(initial_traj_pos_vel)

            initial_traj_pos_vel = torch.stack(traj_pos_vel_l)
            # TODO - now it only accepts 1 goal
            initial_traj_pos_vel = einops.rearrange(initial_traj_pos_vel, 'n h d -> 1 n h d')

            #################################################
            # Fine tune with an optimization-based planner

            # set initial position and velocity trajectory
            self.opt_based_planner.reset(initial_particle_means=initial_traj_pos_vel)

            # Optimize
            trajs_0 = self.opt_based_planner.get_traj()
            trajs_iters = torch.empty((self.opt_based_planner.opt_iters + 1, *trajs_0.shape), **self.tensor_args)
            trajs_iters[0] = trajs_0
            with TimerCUDA() as t_opt_based:
                for i in range(self.opt_based_planner.opt_iters):
                    trajs = self.opt_based_planner.optimize(opt_iters=1, debug=debug, **kwargs)
                    trajs_iters[i + 1] = trajs
            if debug or print_times:
                print(f'Optimization-based Planner -- Optimization time: {t_opt_based.elapsed:.3f} sec')
        if debug or print_times:
            print(f'Hybrid-based Planner -- Optimization time: {t_hybrid.elapsed:.3f} sec')

        # Show the pre-optimization traj and the post-optimization traj
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(initial_traj_pos_vel[0, 0, :, 0].cpu().numpy(), initial_traj_pos_vel[0, 0, :, 1].cpu().numpy(), 'r')
        # ax.plot(trajs_iters[-1][0, :, 0].cpu().numpy(), trajs_iters[-1][0, :, 1].cpu().numpy(), 'b')
        # plt.show()

        if return_iterations:
            return trajs_iters
        else:
            return trajs_iters[-1]
