"""
Multi-Modal Gaussian trajectory prior.
Original source: https://github.com/sashalambert/mpc_trajopt
"""

__author__ = "Alexander Lambert"
__license__ = "MIT"

from copy import copy

import torch
import torch.distributions as dist


class MultiMPPrior:

    def __init__(
            self,
            num_steps,
            dt,
            state_dim,
            dof,
            K_s_inv,
            K_gp_inv,
            start_state,
            means=None,
            K_g_inv=None,
            goal_states=None,
            use_numpy=False,
            tensor_args=None,
    ):
        """
        Motion-Planning prior.

        reference: "Continuous-time Gaussian process motion planning via
        probabilistic inference", Mukadam et al. (IJRR 2018)

        Parameters
        ----------
        num_steps : int
            Planning horizon length (not including start state).
        dt :  float
            Time-step size.
        state_dim : int
            State state_dimension.
        K_s_inv : Tensor
            Start-state inverse covariance. Shape: [state_dim, state_dim]
        K_gp_inv :  Tensor
            Gaussian-process single-step inverse covariance i.e. 'Q_inv'.
            Assumed constant, meaning homoscedastic noise with constant step-size.
            Shape: [2 * state_dim, 2 * state_dim]
        start_state : Tensor
            Shape: [state_dim]
        (Optional) K_g_inv : Tensor
            Goal-state inverse covariance. Shape: [state_dim, state_dim]
        (Optional) goal_state :  Tensor
            Shape: [state_dim]
        (Optional) dof : int
            Degrees of freedom.
        """

        self.state_dim = state_dim
        self.dof = dof
        self.num_steps = num_steps
        self.M = state_dim * (num_steps + 1)
        # self.num_steps = num_steps - 1
        # self.M = state_dim * num_steps
        self.tensor_args = tensor_args
        self.use_numpy = use_numpy

        self.goal_directed = (goal_states is not None)

        if means is None:
            self.num_modes = goal_states.shape[0] if self.goal_directed else 1
            means = self.get_const_vel_mean(
                            start_state,
                            goal_states,
                            dt,
                            num_steps,
                            dof)
        else:
            self.num_modes = means.shape[0]

        # Flatten mean trajectories
        self.means = means.reshape(self.num_modes, -1)

        # TODO - Add different goal Covariances
        # Assume same goal Cov. for now
        Sigma_inv = self.get_const_vel_covariance(
            dt,
            K_s_inv,
            K_gp_inv,
            K_g_inv,
        )

        self.Sigma_inv = Sigma_inv
        self.Sigma_invs = self.Sigma_inv.repeat(self.num_modes, 1, 1)
        self.update_dist(self.means, self.Sigma_invs)

    def update_dist(
            self,
            means,
            Sigma_invs,
    ):
        # Create Multi-variate Normal Distribution
        self.dist = dist.MultivariateNormal(
            means,
            precision_matrix=Sigma_invs,
            # covariance_matrix=torch.inverse(Sigma_invs),
        )

    def get_mean(self, reshape=True):
        if reshape:
            return self.means.clone().detach().reshape(
                self.num_modes, self.num_steps + 1, self.state_dim,
            )
        else:
            self.means.clone().detach()

    def set_mean(self, means_new):
        assert means_new.shape == self.means.shape
        self.means = means_new.clone().detach()
        self.update_dist(self.means, self.Sigma_invs)

    def set_Sigma_invs(self, Sigma_invs_new):
        assert Sigma_invs_new.shape == self.Sigma_invs.shape
        self.Sigma_invs = Sigma_invs_new.clone().detach()
        self.update_dist(self.means, self.Sigma_invs)

    @classmethod
    def const_vel_trajectory(
        cls,
        start_state,
        goal_state,
        dt,
        num_steps,
        dof,
        set_initial_final_vel_to_zero=True,
        tensor_args=None
    ):
        state_traj = torch.zeros(num_steps + 1, 2 * dof, **tensor_args)
        mean_vel = (goal_state[:dof] - start_state[:dof]) / (num_steps * dt)
        for i in range(num_steps + 1):
            state_traj[i, :dof] = start_state[:dof] * (num_steps - i) * 1. / num_steps \
                                  + goal_state[:dof] * i * 1./num_steps
        if set_initial_final_vel_to_zero:
            # initial and final velocities are set to zero
            state_traj[1:-1, dof:] = mean_vel.unsqueeze(0)
        else:
            state_traj[:, dof:] = mean_vel.unsqueeze(0)
        return state_traj

    def get_const_vel_mean(
        self,
        start_state,
        goal_states,
        dt,
        num_steps,
        dof,
    ):

        # Make mean goal-directed if goal_state is provided.
        if self.goal_directed:
            means = []
            for i in range(self.num_modes):
                means.append(self.const_vel_trajectory(
                    start_state,
                    goal_states[i],
                    dt,
                    num_steps,
                    dof,
                    tensor_args=self.tensor_args
                ))
            return torch.stack(means, dim=0)
        else:
            return start_state.repeat(num_steps + 1, 1)

    # def get_const_vel_covariance(
    #     self,
    #     dt,
    #     K_s_inv,
    #     K_gp_inv,
    #     K_g_inv,
    #     precision_matrix=True,
    # ):
    #     # Transition matrix
    #     Phi = torch.eye(self.state_dim, **self.tensor_args)
    #     Phi[:self.dof, self.dof:] = torch.eye(self.dof, **self.tensor_args) * dt
    #     diag_Phis = Phi
    #     for _ in range(self.num_steps - 1):
    #         diag_Phis = torch.block_diag(diag_Phis, Phi)
    #
    #     A = torch.eye(self.M, **self.tensor_args)
    #     A[self.state_dim:, :-self.state_dim] += -1. * diag_Phis
    #     if self.goal_directed:
    #         b = torch.zeros(self.state_dim, self.M,  **self.tensor_args)
    #         b[:, -self.state_dim:] = torch.eye(self.state_dim,  **self.tensor_args)
    #         A = torch.cat((A, b))
    #
    #     Q_inv = K_s_inv
    #     for _ in range(self.num_steps):
    #         Q_inv = torch.block_diag(Q_inv, K_gp_inv).to(**self.tensor_args)
    #     if self.goal_directed:
    #         Q_inv = torch.block_diag(Q_inv, K_g_inv).to(**self.tensor_args)
    #
    #     K_inv = A.t() @ Q_inv @ A
    #     if precision_matrix:
    #         return K_inv
    #     else:
    #         return torch.inverse(K_inv)


    def get_const_vel_covariance(
        self,
        dt,
        K_s_inv,
        K_gp_inv,
        K_g_inv,
        precision_matrix=True,
    ):
        # To construct the covariance we need to use float64, because the small covariances of the start, GP and goal
        # state distributions lead to very large inverse covariances during the construction of these matrices.

        tensor_args_tmp = copy(self.tensor_args)
        tensor_args_tmp['dtype'] = torch.float64

        # Transition matrix
        Phi = torch.eye(self.state_dim, **tensor_args_tmp)
        Phi[:self.dof, self.dof:] = torch.eye(self.dof, **tensor_args_tmp) * dt
        diag_Phis = Phi
        for _ in range(self.num_steps - 1):
            diag_Phis = torch.block_diag(diag_Phis, Phi)

        A = torch.eye(self.M, **tensor_args_tmp)
        A[self.state_dim:, :-self.state_dim] += -1. * diag_Phis
        if self.goal_directed:
            b = torch.zeros(self.state_dim, self.M,  **tensor_args_tmp)
            b[:, -self.state_dim:] = torch.eye(self.state_dim,  **tensor_args_tmp)
            A = torch.cat((A, b))

        Q_inv = K_s_inv
        for _ in range(self.num_steps):
            Q_inv = torch.block_diag(Q_inv, K_gp_inv).to(**tensor_args_tmp)
        if self.goal_directed:
            Q_inv = torch.block_diag(Q_inv, K_g_inv).to(**tensor_args_tmp)

        K_inv = A.t() @ Q_inv @ A
        if precision_matrix:
            return K_inv.to(**self.tensor_args)
        else:
            return torch.inverse(K_inv).to(**self.tensor_args)

    def sample(self, num_samples):
        return self.dist.sample((num_samples,)).view(
            num_samples, self.num_modes, self.num_steps + 1, self.state_dim,
        ).transpose(1, 0)

    def log_prob(self, x):
        return self.dist.log_prob(x)
