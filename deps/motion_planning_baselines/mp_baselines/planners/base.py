import abc
from abc import ABC, abstractmethod
from typing import Tuple
import torch

from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor
from torch_robotics.trajectory.utils import finite_difference_vector


class MPPlanner(ABC):
    """Base class for all planners."""

    def __init__(
        self,
        name: str = None,
        tensor_args: dict = None,
        **kwargs
    ):
        self.name = name
        self.tensor_args = tensor_args
        self._kwargs = kwargs

    @abstractmethod
    def optimize(self, opt_iters: int = 1, **observation) -> Tuple[bool, torch.Tensor]:
        """Plan a path from start to goal.

        Args:
            opt_iters: Number of optimization iters.
            observation: dict of observations.

        Returns:
            success: True if a path was found.
            path: Path from start to goal.
        """
        pass

    def __call__(self, opt_iters: int = 1, **observation) -> Tuple[bool, torch.Tensor]:
        """Plan a path from start to goal.

        Args:
            start: Start position.
            goal: Goal position.

        Returns:
            success: True if a path was found.
            path: Path from start to goal.
        """
        return self.optimize(opt_iters, **observation)

    def __repr__(self):
        return f"{self.name}({self._kwargs})"

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError


class OptimizationPlanner(MPPlanner):

    def __init__(
        self,
        name: str = 'OptimizationPlanner',
        n_dof: int = None,
        n_support_points: int = None,
        n_interpolated_points: int = None,
        num_particles_per_goal: int = None,
        opt_iters: int = None,
        dt: float = None,
        start_state: torch.Tensor = None,
        cost=None,
        initial_particle_means=None,
        multi_goal_states: torch.Tensor = None,
        sigma_start_init: float = 0.001,
        sigma_goal_init: float = 0.001,
        sigma_gp_init: float = 10.,
        pos_only: bool = False,
        tensor_args: dict = None, **kwargs
    ):
        super().__init__(name, tensor_args, **kwargs)
        self.n_dof = n_dof
        self.dim = 2 * self.n_dof
        self.n_support_points = n_support_points
        self.n_interpolated_points = n_interpolated_points
        self.num_particles_per_goal = num_particles_per_goal
        self.opt_iters = opt_iters
        self.dt = dt
        self.pos_only = pos_only

        self.start_state = start_state
        self.multi_goal_states = multi_goal_states
        if multi_goal_states is None:  # NOTE(an): if there is no goal, we assume xere is at least one solution
            self.num_goals = 1
        else:
            assert multi_goal_states.ndim == 2
            self.num_goals = multi_goal_states.shape[0]
        self.num_particles = self.num_goals * self.num_particles_per_goal
        self.cost = cost
        self.initial_particle_means = initial_particle_means
        self._particle_means = None
        if self.pos_only:
            self.d_state_opt = self.n_dof
            self.start_state = self.start_state
        else:
            self.d_state_opt = 2 * self.n_dof
            self.start_state = torch.cat([self.start_state, torch.zeros_like(self.start_state)], dim=-1)
            if self.multi_goal_states is not None:
                self.multi_goal_states = torch.cat([self.multi_goal_states, torch.zeros_like(self.multi_goal_states)], dim=-1)

        self.sigma_start_init = sigma_start_init
        self.sigma_goal_init = sigma_goal_init
        self.sigma_gp_init = sigma_gp_init

    def get_GP_prior(
            self,
            start_K,
            gp_K,
            goal_K,
            state_init,
            particle_means=None,
            goal_states=None,
            tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = self.tensor_args
        return MultiMPPrior(
            self.n_support_points - 1,
            self.dt,
            self.dim,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,  # NOTE(sasha) Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
            tensor_args=tensor_args,
        )

    def const_vel_trajectories(
        self,
        start_state,
        multi_goal_states,
    ):
        traj_dim = (multi_goal_states.shape[0], self.n_support_points, self.dim)
        state_traj = torch.zeros(traj_dim, **self.tensor_args)
        mean_vel = (multi_goal_states[:, :self.n_dof] - start_state[:, :self.n_dof]) / (self.n_support_points * self.dt)
        for i in range(self.n_support_points):
            state_traj[:, i, :self.n_dof] = start_state[:, :self.n_dof] * (self.n_support_points - i - 1) / (self.n_support_points - 1) \
                                  + multi_goal_states[:, :self.n_dof] * i / (self.n_support_points - 1)
        state_traj[:, :, self.n_dof:] = mean_vel.unsqueeze(1).repeat(1, self.n_support_points, 1)
        return state_traj

    def get_random_trajs(self):
        # force torch.float64
        tensor_args = dict(dtype=torch.float64, device=self.tensor_args['device'])
        # set zero velocity for GP prior
        start_state = torch.cat((self.start_state, torch.zeros_like(self.start_state)), dim=-1).to(**tensor_args)
        if self.multi_goal_states is not None:
            multi_goal_states = torch.cat((self.multi_goal_states, torch.zeros_like(self.multi_goal_states)), dim=-1).to(**tensor_args)
        else:
            multi_goal_states = None
        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.n_dof * 2,
            self.sigma_start_init,
            start_state,
            tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.n_support_points - 1,
            tensor_args,
        )

        self.multi_goal_prior_init = []
        if multi_goal_states is not None:
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.n_dof * 2,
                        self.sigma_goal_init,
                        multi_goal_states[i],
                        tensor_args,
                    )
                )
        self._traj_dist = self.get_GP_prior(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if multi_goal_states is not None else None,
                start_state,
                goal_states=multi_goal_states,
                tensor_args=tensor_args,
            )
        particles = self._traj_dist.sample(self.num_particles_per_goal).to(**tensor_args)
        self.traj_dim = particles.shape
        del self._traj_dist  # free memory
        return particles.flatten(0, 1).to(**self.tensor_args)

    def _get_traj(self):
        """
            Get position-velocity trajectory from control distribution.
        """
        trajs = self._particle_means.clone()
        if self.pos_only:
            # Linear velocity by central finite differencing
            vels = finite_difference_vector(trajs, dt=self.dt)
            trajs = torch.cat((trajs, vels), dim=1)
        return trajs

    def get_traj(self):
        return self._get_traj()

    def _get_costs(self, state_trajectories, **observation):
        if self.cost is None:
            costs = torch.zeros(self.num_particles, )
        else:
            costs = self.cost(state_trajectories, **observation)
        return costs

    def render(self, ax, **kwargs):
        raise NotImplementedError
