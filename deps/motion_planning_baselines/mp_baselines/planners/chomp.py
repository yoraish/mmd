import torch

from mp_baselines.planners.base import OptimizationPlanner
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import batched_weighted_dot_prod


class CHOMP(OptimizationPlanner):
    # https://www.ri.cmu.edu/pub_files/2009/5/icra09-chomp.pdf

    def __init__(
            self,
            n_dof: int,
            n_support_points: int,
            num_particles_per_goal: int,
            opt_iters: int,
            dt: float,
            start_state: torch.Tensor,
            cost=None,
            weight_prior_cost=0.1,
            initial_particle_means=None,
            step_size: float = 1.,
            grad_clip: float = .01,
            multi_goal_states: torch.Tensor = None,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=10.,
            pos_only: bool = True,
            **kwargs
    ):
        super(CHOMP, self).__init__(name='CHOMP',
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
                                    **kwargs)

        # CHOMP params
        self.lr = step_size
        self.grad_clip = grad_clip

        self._particle_means = None
        # Precision matrix, shape: [ctrl_dim, n_support_points, n_support_points]
        self.Sigma_inv = self._get_R_mat(dt=self.dt, n_support_points=self.n_support_points, tensor_args=self.tensor_args)
        self.Sigma = torch.inverse(self.Sigma_inv)
        self.reset(initial_particle_means=initial_particle_means)

        # Weight prior
        self.weight_prior_cost = weight_prior_cost

    def _get_R_mat2(self):
        """
        STOMP time-correlated Precision matrix.
        Central finite difference velocity.
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
        A_mat = A_mat * 1./self.dt**2
        R_mat = A_mat.t() @ A_mat
        return R_mat.to(**self.tensor_args)

    @classmethod
    def _get_R_mat(
            cls,
            dt=0.01,
            n_support_points=64,
            tensor_args=None,
            **kwargs
    ):
        """
        CHOMP time-correlated Precision matrix.
        Backward finite difference velocity.
        """
        lower_diag = -torch.diag(torch.ones(n_support_points - 1), diagonal=-1)
        diag = 1 * torch.eye(n_support_points)
        K_mat = diag + lower_diag
        K_mat = torch.cat((K_mat, torch.zeros(1, n_support_points)), dim=0)
        K_mat[-1, -1] = -1.
        K_mat = K_mat * 1. / dt ** 2
        R_mat = K_mat.t() @ K_mat

        return R_mat.to(**tensor_args)

    def reset(
            self,
            initial_particle_means=None,
    ):
        # Straight line position-trajectory from start to goal with const vel
        if initial_particle_means is not None:
            self._particle_means = initial_particle_means.clone()
        else:
            self._particle_means = self.get_random_trajs()

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """

        self._run_optimization(opt_iters, **observation)
        # get mean trajectory
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
            self._particle_means.requires_grad_(True)
            costs = self._eval(self._particle_means, **observation)

            # Get grad
            costs.sum().backward(retain_graph=True)
            # For stabilizing and preventing high gradients
            self._particle_means.grad.data.clamp_(-self.grad_clip, self.grad_clip)
            # zeroing grad at start and goal
            self._particle_means.grad.data[..., 0, :] = 0.
            self._particle_means.grad.data[..., -1, :] = 0.

            # Update trajectory
            self._particle_means.data.add_(-self.lr * self._particle_means.grad.data)
            self._particle_means.grad.detach_()
            self._particle_means.grad.zero_()

        self._particle_means = self._particle_means.detach()

    def _eval(self, x, **observation):
        """
        Evaluate costs.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # Evaluate (collision, ...) costs
        costs = self._get_costs(x, **observation)

        # Add smoothness term (prior)
        R_mat = self.Sigma_inv
        smooth_cost = batched_weighted_dot_prod(x, R_mat, x).sum()

        costs += self.weight_prior_cost * smooth_cost

        return costs
