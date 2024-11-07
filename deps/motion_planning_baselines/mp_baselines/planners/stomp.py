import einops
import torch
import torch.distributions as dist

from mp_baselines.planners.base import OptimizationPlanner


class STOMP(OptimizationPlanner):

    def __init__(
            self,
            n_dof: int,
            n_support_points: int,
            num_particles_per_goal: int,
            num_samples: int,
            opt_iters: int,
            dt: float,
            start_state: torch.Tensor,
            cost=None,
            initial_particle_means=None,
            multi_goal_states: torch.Tensor = None,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=10.,
            temperature: float = 1.,
            step_size: float = 1.,
            sigma_spectral: float = 0.1,
            goal_state: torch.Tensor = None,
            pos_only: bool = True,
            tensor_args: dict = None,
            **kwargs
    ):
        super(STOMP, self).__init__(name='STOMP',
                                    n_dof=n_dof,
                                    n_support_points=n_support_points,
                                    num_particles_per_goal=num_particles_per_goal,
                                    opt_iters=opt_iters,
                                    dt=dt,
                                    start_state=start_state,
                                    cost=cost,
                                    initial_particle_means=initial_particle_means,
                                    multi_goal_states=multi_goal_states,
                                    sigma_start_init=sigma_start_init,
                                    sigma_goal_init=sigma_goal_init,
                                    sigma_gp_init=sigma_gp_init,
                                    pos_only=pos_only,
                                    tensor_args=tensor_args)

        # STOMP params
        self.lr = step_size
        self.sigma_spectral = sigma_spectral

        self.start_state = start_state
        self.goal_state = goal_state
        self.num_samples = num_samples
        self.temperature = temperature

        self._particle_means = None
        self._weights = None
        self._sample_dist = None

        # Precision matrix, shape: [ctrl_dim, n_support_points, n_support_points]
        self.Sigma_inv = self._get_R_mat()
        self.Sigma = torch.inverse(self.Sigma_inv)
        self.reset(initial_particle_means=initial_particle_means)
        self.best_cost = torch.inf

    def _get_R_mat(self):
        """
        STOMP time-correlated Precision matrix.
        """
        upper_diag = torch.diag(torch.ones(self.n_support_points - 1), diagonal=1)
        lower_diag = torch.diag(torch.ones(self.n_support_points - 1), diagonal=-1,)
        diag = -2 * torch.eye(self.n_support_points)
        A_mat = upper_diag + diag + lower_diag
        A_mat = torch.cat(
            (torch.zeros(1, self.n_support_points),
             A_mat,
             torch.zeros(1, self.n_support_points)),
            dim=0,
        )
        A_mat[0, 0] = 1.
        A_mat[-1, -1] = 1.
        A_mat = A_mat * 1./self.dt**2 * self.sigma_spectral
        R_mat = A_mat.t() @ A_mat
        return R_mat.to(**self.tensor_args)

    def set_noise_dist(self):
        """
            Additive Gaussian noise distribution over 1-dimensional trajectory.
        """
        self._noise_dist = dist.MultivariateNormal(
            torch.zeros(self.num_particles, self.n_support_points, **self.tensor_args),
            precision_matrix=self.Sigma_inv,
        )

    def sample(self):
        """
            Generate trajectory samples from Gaussian control dist.
            Return: position-trajectory samples, of shape: [num_particles, num_samples, n_support_points, n_dof]
        """
        noise = self._noise_dist.sample((self.num_samples, self.d_state_opt)).transpose(0, 2).transpose(1, 3).transpose(1, 2)  # [num_particles, num_samples, n_support_points, n_dof]

        # Force Bound to zero ##
        noise[..., -1, :] = 0
        noise[..., 0, :] = 0
        samples = self._particle_means.unsqueeze(1) + noise
        return samples

    def reset(
            self,
            initial_particle_means=None,
    ):
        # Straightline position-trajectory from start to goal
        if initial_particle_means is not None:
            self._particle_means = initial_particle_means.clone()
        else:
            self._particle_means = self.get_random_trajs()
        self.set_noise_dist()
        self.state_particles = self.sample()

    def const_vel_trajectory(
        self,
        start_state,
        goal_state,
    ):
        num_steps = self.n_support_points - 1
        state_traj = torch.zeros(num_steps + 1, self.d_state_opt, **self.tensor_args)
        for i in range(num_steps + 1):
            state_traj[i, :self.n_dof] = start_state[:self.n_dof] * (num_steps - i) * 1. / num_steps \
                                  + goal_state[:self.n_dof] * i * 1./num_steps
        if not self.pos_only:
            mean_vel = (goal_state[:self.n_dof] - start_state[:self.n_dof]) / (num_steps * self.dt)
            state_traj[:, self.n_dof:] = mean_vel.unsqueeze(0)
        return state_traj

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """
        self._run_optimization(opt_iters, **observation)
        # get best trajectory
        curr_traj = self._get_traj()
        return curr_traj

    def _run_optimization(self, opt_iters, **observation):
        """
            Run optimization iterations.
        """
        optim_vis = observation.get('optim_vis', False)
        if opt_iters is None:
            opt_iters = self.opt_iters
        for opt_step in range(opt_iters):
            self.costs = self._sample_and_eval(**observation)
            self._update_distribution(self.costs, self.state_particles)
            self._particle_means = self._particle_means.detach()

    def _sample_and_eval(self, **observation):
        """
            Sample trajectories from distribution and evaluate costs.
        """
        # Sample state-trajectory particles
        self.state_particles = self.sample()

        ## Optional : Clamp for joint limits
        # if isinstance(self.max_control, list):
        #     for d in range(self.control_dim):
        #         control_samples[..., d] = control_samples[..., d].clamp(
        #             min=-self.max_control[d],
        #             max=self.max_control[d],
        #         )
        # else:
        #     control_samples = control_samples.clamp(
        #         min=-self.max_control,
        #         max=self.max_control,
        #     )

        # Evaluate quadratic costs
        costs = self._get_costs(self.state_particles.flatten(0, 1), **observation)
        costs = einops.rearrange(costs, "(p s) -> p s", p=self.num_particles, s=self.num_samples)

        ## Optional : Add cost term from importance sampling ratio
        # for i in range(self.control_dim):
        #     Sigma_inv = self.ctrl_dist.Sigma[..., i].inverse()
        #     V = self.state_particles[..., i]
        #     U = self.U_mean[..., i]
        #     costs += self.lambda_ * (V @ Sigma_inv @ U).reshape(-1, 1)

        # Add smoothness term
        # smooth_cost = self.state_particles.transpose(1, 2) @ self.Sigma_inv.unsqueeze(0) @ self.state_particles
        # costs += smooth_cost.diagonal(dim1=1, dim2=2).sum(-1)

        return costs

    def _update_distribution(self, costs, traj_particles):
        """
            Get sample weights and pdate trajectory mean.
        """
        self._weights = self._calc_sample_weights(costs)
        self._weights = self._weights.reshape(self.num_particles, self.num_samples, 1, 1)

        ## Optional: STOMP Covariance-weighted update
        self._particle_means.add_(
            self.lr * self.Sigma @ (
                self._weights * (traj_particles - self._particle_means.unsqueeze(1))
            ).sum(1)
        )

        # self._particle_means.add_(
        #     self.lr * (
        #         self._weights * (traj_particles - self._particle_means.unsqueeze(1))
        #     ).sum(0)
        # )

    def _calc_sample_weights(self, costs):
        return torch.softmax(-costs / self.temperature, dim=1)  # num_samples dim
