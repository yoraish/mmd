import torch
from mp_baselines.planners.priors.gaussian import get_multivar_gaussian_prior
from mp_baselines.planners.base import MPPlanner


class MPPI(MPPlanner):

    def __init__(
            self,
            system,
            num_ctrl_samples,
            rollout_steps,
            opt_iters,
            control_std=None,
            initial_mean=None,
            step_size=1.,
            temp=1.,
            cov_prior_type='indep_ctrl',
            tensor_args=None,
            **kwargs
    ):
        super(MPPI, self).__init__(name='MPPI', tensor_args=tensor_args)
        self.system = system
        self.state_dim = system.state_dim
        self.control_dim = system.control_dim
        self.rollout_steps = rollout_steps
        self.num_ctrl_samples = num_ctrl_samples
        self.opt_iters = opt_iters

        self.step_size = step_size
        self.temp = temp
        self._mean = torch.zeros(
            self.rollout_steps,
            self.control_dim,
            **self.tensor_args,
        )
        self.control_std = control_std
        self.cov_prior_type = cov_prior_type
        self.weights = None

        self.ctrl_dist = get_multivar_gaussian_prior(
                control_std,
                rollout_steps,
                self.control_dim,
                Cov_type=cov_prior_type,
                mu_init=self._mean,
                tensor_args=self.tensor_args,
            )

        Cov_inv = []
        for i in range(self.control_dim):
            Cov_inv.append(self.ctrl_dist.Cov[..., i].inverse())
        self.Cov_inv = torch.stack(Cov_inv)
        self.best_cost = torch.inf
        self.reset(initial_mean=initial_mean)

    def reset(self, initial_mean=None):
        if initial_mean is not None:
            self._mean = initial_mean.clone()
        else:
            self._mean = torch.zeros(
                self.rollout_steps,
                self.control_dim,
                **self.tensor_args,
            )
        self.update_ctrl_dist()

    def update_ctrl_dist(self):
        # Update mean
        self.ctrl_dist.update_means(self._mean)

    def update_controller(self, costs, U_sampled):
        weights = torch.softmax(
            -costs / self.temp,
            dim=0,
        )
        self.weights = weights.clone()

        weights = weights.reshape(-1, 1, 1)
        self._mean.add_(
            self.step_size * (
                weights * (U_sampled - self._mean.unsqueeze(0))
            ).sum(0)
        )

        self.update_ctrl_dist()

    def sample_and_eval(self, **observation):
        # state_trajectories = torch.empty(
        #     self.num_ctrl_samples,
        #     self.rollout_steps, # self.rollout_steps + 1,
        #     self.state_dim,
        #     **self.tensor_args,
        # )
        #
        # # Sample control sequences
        # control_samples = self.ctrl_dist.sample(self.num_ctrl_samples)
        #
        # # Roll-out dynamics
        # state_trajectories[:, 0] = observation['state']
        # for i in range(self.rollout_steps - 1):
        #     state_trajectories[:, i+1] = self.system.dynamics(
        #         state_trajectories[:, i].unsqueeze(1),
        #         control_samples[:, i].unsqueeze(1),
        #     ).squeeze(1)
        # self.state_trajectories = state_trajectories.clone()


        # Sample control sequences
        control_samples = self.ctrl_dist.sample(self.num_ctrl_samples)
        # Rollout dynamics
        state_trajectories = self.get_state_trajectories_rollout(
            controls=control_samples, num_ctrl_samples=self.num_ctrl_samples, **observation
        )
        self.state_trajectories = state_trajectories.clone()

        # Evaluate costs
        self.costs = self.system.traj_cost(
            state_trajectories.transpose(0, 1).unsqueeze(2),
            control_samples.transpose(0, 1).unsqueeze(2),
            **observation
        )

        # Add cost from importance-sampling ratio
        for i in range(self.control_dim):
            V = control_samples[..., i]
            U = self._mean[..., i]
            self.costs += self.temp * (V @ self.Cov_inv[i] @ U).reshape(-1, 1)

        return (
            control_samples,
            state_trajectories,
            self.costs
        )

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            with torch.no_grad():
                (control_samples,
                 state_trajectories,
                 costs) = self.sample_and_eval(**observation)
                self._save_best()
                self.update_controller(costs, control_samples)
                self._mean = self._mean.detach()

        self._recent_control_samples = control_samples
        self._recent_state_trajectories = state_trajectories
        self._recent_weights = self.weights

        return (
            control_samples,
            state_trajectories,
            costs
        )
    
    def _save_best(self):
        best_cost = torch.min(self.costs)
        idx = torch.argmin(self.costs)
        if best_cost < self.best_cost:
            self.best_cost = best_cost
            self.best_traj = self.state_trajectories[idx,...]

    def pop(self):
        action = self._mean[0, :].clone().detach()
        self.shift()
        return action

    def shift(self):
        self._mean = self._mean.roll(shifts=-1, dims=-1)
        self._mean[-1:] = 0.

    def get_recent_samples(self):
        return (
            self._recent_control_samples.detach().clone(),
            self._recent_state_trajectories.detach().clone(),
            self._recent_weights.detach().clone()
        )

    def get_mean_controls(self):
        return self._mean

    def get_state_trajectories_rollout(self, controls=None, num_ctrl_samples=None, **observation):
        state_trajectories = torch.empty(
            1 if num_ctrl_samples is None else num_ctrl_samples,
            self.rollout_steps,  # self.rollout_steps + 1,
            self.state_dim,
            **self.tensor_args,
        )

        if controls is None:
            control_samples = self._mean.unsqueeze(0)
        else:
            control_samples = controls

        # Roll-out dynamics
        state_trajectories[:, 0] = observation['state']
        for i in range(self.rollout_steps - 1):
            state_trajectories[:, i+1] = self.system.dynamics(
                state_trajectories[:, i].unsqueeze(1),
                control_samples[:, i].unsqueeze(1),
            ).squeeze(1)
        return state_trajectories.clone()

    def render(self, ax, **kwargs):
        raise NotImplementedError
