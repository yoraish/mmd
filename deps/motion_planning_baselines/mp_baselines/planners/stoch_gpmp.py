# Adapted from https://github.com/anindex/stoch_gpmp

import time
from copy import copy

import torch

from mp_baselines.planners.base import OptimizationPlanner
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor
from mp_baselines.planners.gpmp2 import build_gpmp2_cost_composite


class StochGPMP(OptimizationPlanner):

    def __init__(
            self,
            robot=None,
            n_dof: int = None,
            n_support_points: int = None,
            num_particles_per_goal: int = None,
            opt_iters: int = None,
            dt: float = None,
            start_state: torch.Tensor = None,
            step_size=1.,
            multi_goal_states=None,
            initial_particle_means=None,
            sigma_start_init=None,
            sigma_start_sample=None,
            sigma_goal_init=None,
            sigma_goal_sample=None,
            sigma_gp_init=None,
            sigma_gp_sample=None,
            num_samples=2,
            temperature=1.,
            **kwargs
    ):
        super(StochGPMP, self).__init__(
            name='StochGPMP',
            n_dof=n_dof,
            n_support_points=n_support_points,
            num_particles_per_goal=num_particles_per_goal,
            opt_iters=opt_iters,
            dt=dt,
            start_state=start_state,
            initial_particle_means=initial_particle_means,
            multi_goal_states=multi_goal_states,
            sigma_start_init=sigma_start_init,
            sigma_goal_init=sigma_goal_init,
            sigma_gp_init=sigma_gp_init,
            pos_only=False,
            **kwargs
        )

        self.d_state_opt = 2 * self.n_dof

        self.goal_directed = (multi_goal_states is not None)
        if not self.goal_directed:  # TODO NOTE(an): if xere is no goal, we assume xere is at least one solution
            self.num_goals = 1
        else:
            assert multi_goal_states.dim() == 2
            self.num_goals = multi_goal_states.shape[0]

        self.num_samples = num_samples
        self.step_size = step_size
        self.temperature = temperature
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal_sample = sigma_goal_sample
        self.sigma_gp_sample = sigma_gp_sample

        self._mean = None
        self._weights = None
        self._sample_dist = None

        ##############################################
        # Construct cost function
        self.cost = build_gpmp2_cost_composite(
            robot=robot,
            n_support_points=n_support_points,
            dt=dt,
            start_state=start_state,
            multi_goal_states=multi_goal_states,
            num_particles_per_goal=num_particles_per_goal,
            num_samples=num_samples,
            **kwargs
        )

        ##############################################
        # Initialize particles
        self.reset(initial_particle_means=initial_particle_means)

    def reset(
            self,
            start_state=None,
            multi_goal_states=None,
            initial_particle_means=None,
    ):
        if start_state is not None:
            self.start_state = start_state.detach().clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.detach().clone()

        self.set_prior_factors()

        if initial_particle_means is not None:
            if initial_particle_means == 'const_vel':
                self._particle_means = self.const_vel_trajectories(
                    self.start_state,
                    self.multi_goal_states,
                )
            else:
                self._particle_means = initial_particle_means
        else:
            # Initialization particles from prior distribution
            self._init_dist = self.get_prior_dist(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if self.goal_directed else None,
                self.start_state,
                goal_states=self.multi_goal_states,
            )
            self._particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
            del self._init_dist  # free memory
        self._particle_means = self._particle_means.flatten(0, 1)

        # Sampling distributions
        self._sample_dist = self.get_prior_dist(
            self.start_prior_sample.K,
            self.gp_prior_sample.Q_inv[0],
            self.multi_goal_prior_sample[0].K if self.goal_directed else None,
            self.start_state,
            particle_means=self._particle_means,
            goal_states=self.multi_goal_states
        )
        self.Sigma_inv = self._sample_dist.Sigma_inv.to(**self.tensor_args)
        self.state_samples = self._sample_dist.sample(self.num_samples).to(**self.tensor_args)

    def set_prior_factors(self):
        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_init,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_init = []
        if self.goal_directed:
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_init,    # NOTE(sasha) Assume same goal Cov. for now
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

        #========= Sampling factors ===============
        self.start_prior_sample = UnaryFactor(
            self.d_state_opt,
            self.sigma_start_sample,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior_sample = GPFactor(
            self.n_dof,
            self.sigma_gp_sample,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

        self.multi_goal_prior_sample = []
        if self.goal_directed:
            for i in range(self.num_goals):
                self.multi_goal_prior_sample.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_sample,   # NOTE(sasha) Assume same goal Cov. for now
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

    def const_vel_trajectories(
        self,
        start_state,
        multi_goal_states,
    ):
        traj_dim = (multi_goal_states.shape[0], self.num_particles_per_goal, self.n_support_points, self.d_state_opt)
        state_traj = torch.zeros(traj_dim, **self.tensor_args)
        mean_vel = (multi_goal_states[:, :self.n_dof] - start_state[:self.n_dof]) / (self.n_support_points * self.dt)
        for i in range(self.n_support_points):
            interp_state = start_state[:self.n_dof] * (self.n_support_points - i - 1) / (self.n_support_points - 1) \
                                  + multi_goal_states[:, :self.n_dof] * i / (self.n_support_points - 1)
            state_traj[:, :, i, :self.n_dof] = interp_state.unsqueeze(1)
        state_traj[:, :, :, self.n_dof:] = mean_vel.unsqueeze(1).unsqueeze(1)
        return state_traj

    def get_prior_dist(
            self,
            start_K,
            gp_K,
            goal_K,
            state_init,
            particle_means=None,
            goal_states=None,
    ):
        return MultiMPPrior(
            self.n_support_points - 1,
            self.dt,
            2 * self.n_dof,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,  # Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
            tensor_args=self.tensor_args
        )

    def _get_costs(self, **observation):
        costs = self.cost.eval(self.state_samples, **observation).reshape(self.num_particles, self.num_samples)

        # Add cost from importance-sampling ratio
        V = self.state_samples.view(-1, self.num_samples, self.n_support_points * self.d_state_opt)  # flatten trajectories
        U = self._particle_means.view(-1, 1, self.n_support_points * self.d_state_opt)
        costs += self.temperature * (V @ self.Sigma_inv @ U.transpose(1, 2)).squeeze(2)
        return costs

    def sample_and_eval(self, **observation):
        # TODO: update prior covariance with new goal location

        # Sample state-trajectory particles
        self.state_samples = self._sample_dist.sample(self.num_samples).to(**self.tensor_args)

        # Evaluate costs
        costs = self._get_costs(**observation)

        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]

        position_seq_mean = self._particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self._particle_means[..., -self.n_dof:].clone()

        return (
            velocity_seq,
            position_seq,
            velocity_seq_mean,
            position_seq_mean,
            costs,
        )

    def _update_distribution(self, costs, traj_samples):
        self._weights = torch.softmax(-costs / self.temperature, dim=1)
        self._weights = self._weights.reshape(-1, self.num_samples, 1, 1)

        # sum over particles
        approx_grad = (self._weights * (traj_samples - self._particle_means.unsqueeze(1))).sum(1)

        self._particle_means.add_(
            self.step_size * approx_grad
        )
        self._sample_dist.set_mean(self._particle_means.view(self.num_particles, -1))

        return approx_grad

    def optimize(
            self,
            opt_iters=None,
            debug=False,
            **observation
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            with torch.no_grad():
                (control_samples,
                 state_trajectories,
                 control_particles,
                 state_particles,
                 costs,) = self.sample_and_eval(**observation)

                approx_grad = self._update_distribution(costs, self.state_samples)

        self._recent_control_samples = control_samples
        self._recent_control_particles = control_particles
        self._recent_state_trajectories = state_trajectories
        self._recent_state_particles = state_particles
        self._recent_weights = self._weights

        # get mean trajectory
        curr_traj = self._get_traj()
        return curr_traj

    # def _get_traj(self, mode='best'):
    #     if mode == 'best':
    #         # TODO: Fix for multi-particles
    #         particle_ind = self._weights.argmax()
    #         traj = self.state_samples[particle_ind].clone()
    #     elif mode == 'mean':
    #         traj = self._mean.clone()
    #     else:
    #         raise ValueError('Unidentified sampling mode in get_next_action')
    #     return traj

    def get_recent_samples(self):
        return (
            self._recent_state_trajectories.detach().clone(),
            self._recent_state_particles.detach().clone(),
            self._recent_control_samples.detach().clone(),
            self._recent_control_particles.detach().clone(),
            self._recent_weights.detach().clone(),
        )

    def sample_trajectories(self, num_samples_per_particle):
        self._sample_dist.set_mean(self._particle_means.view(self.num_particles, -1))
        self.state_samples = self._sample_dist.sample(num_samples_per_particle).to(
            **self.tensor_args)
        position_seq = self.state_samples[..., :self.n_dof]
        velocity_seq = self.state_samples[..., -self.n_dof:]
        return (
            position_seq,
            velocity_seq,
        )
