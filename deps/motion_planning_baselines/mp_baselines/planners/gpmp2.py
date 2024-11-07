# Adapted from https://github.com/anindex/stoch_gpmp
from copy import copy

import einops
import numpy as np
import torch

from mp_baselines.planners.base import OptimizationPlanner
from mp_baselines.planners.costs.cost_functions import CostGP, CostGoalPrior, CostCollision, CostComposite
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor

try:
    import cholespy
except ModuleNotFoundError:
    pass

from torch_robotics.torch_utils.torch_timer import TimerCUDA


def build_gpmp2_cost_composite(
    robot=None,
    n_support_points=None,
    dt=None,
    start_state=None,
    multi_goal_states=None,
    num_particles_per_goal=None,
    collision_fields=None,
    extra_costs=[],
    sigma_start=1e-5,
    sigma_gp=1e-2,
    sigma_coll=1e-5,
    sigma_goal_prior=1e-5,
    num_samples: int = 64,
    tensor_args=None,
    **kwargs,
):
    """
    Construct cost composite function for GPMP and StochGPMP
    """
    cost_func_list = []

    # Start state + GP cost
    cost_sigmas = dict(
        sigma_start=sigma_start,
        sigma_gp=sigma_gp,
    )
    start_state_zero_vel = torch.cat((start_state, torch.zeros(start_state.nelement(), **tensor_args)))
    cost_gp_prior = CostGP(
        robot, n_support_points, start_state_zero_vel, dt,
        cost_sigmas,
        tensor_args=tensor_args
    )
    cost_func_list.append(cost_gp_prior)

    # Goal state cost
    if multi_goal_states is not None:
        multi_goal_states_zero_vel = torch.cat((multi_goal_states, torch.zeros_like(multi_goal_states)),
                                               dim=-1).unsqueeze(0)  # add batch dim for interface
        cost_goal_prior = CostGoalPrior(
            robot, n_support_points, multi_goal_states=multi_goal_states_zero_vel,
            num_particles_per_goal=num_particles_per_goal,
            num_samples=num_samples,
            sigma_goal_prior=sigma_goal_prior,
            tensor_args=tensor_args
        )
        cost_func_list.append(cost_goal_prior)

    # Collision costs
    for collision_field in collision_fields:
        cost_collision = CostCollision(
            robot, n_support_points,
            field=collision_field,
            sigma_coll=sigma_coll,
            tensor_args=tensor_args
        )
        cost_func_list.append(cost_collision)

    # Other costs
    if extra_costs:
        cost_func_list.append(*extra_costs)

    cost_composite = CostComposite(
        robot, n_support_points, cost_func_list,
        tensor_args=tensor_args
    )
    return cost_composite


class GPMP2(OptimizationPlanner):

    def __init__(
            self,
            robot=None,
            n_dof: int = None,
            n_support_points: int = None,
            n_interpolated_points: int = None,
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
            solver_params=None,
            stop_criteria=None,  # or None
            **kwargs
    ):
        super(GPMP2, self).__init__(
            name='GPMP',
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
            robot=None,
            **kwargs
        )

        self.d_state_opt = 2 * self.n_dof

        self.goal_directed = (multi_goal_states is not None)
        if not self.goal_directed:
            self.num_goals = 1
        else:
            assert multi_goal_states.dim() == 2
            self.num_goals = multi_goal_states.shape[0]

        self.step_size = step_size
        self.sigma_start_sample = sigma_start_sample
        self.sigma_goal_sample = sigma_goal_sample

        self.solver_params = solver_params

        self.N = self.d_state_opt * self.n_support_points

        self._mean = None
        self._weights = None
        self._dist = None

        self.stop_criteria = stop_criteria

        self.robot = robot

        ##############################################
        # Construct cost function
        self.cost = build_gpmp2_cost_composite(
            robot=robot,
            n_support_points=n_support_points,
            dt=dt,
            start_state=start_state,
            multi_goal_states=multi_goal_states,
            num_particles_per_goal=num_particles_per_goal,
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
            self.start_state = start_state.clone()

        if multi_goal_states is not None:
            self.multi_goal_states = multi_goal_states.clone()

        self.set_prior_factors()

        # Initialization particles from prior distribution
        if initial_particle_means is not None:
            self._particle_means = initial_particle_means
        else:
            self._init_dist = self.get_dist(
                self.start_prior_init.K,
                self.gp_prior_init.Q_inv[0],
                self.multi_goal_prior_init[0].K if self.goal_directed else None,
                self.start_state,
                goal_states=self.multi_goal_states,
            )
            self._particle_means = self._init_dist.sample(self.num_particles_per_goal).to(**self.tensor_args)
            del self._init_dist  # freeing memory
        self._particle_means = self._particle_means.flatten(0, 1)

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

        if self.goal_directed:
            self.multi_goal_prior_init = []
            for i in range(self.num_goals):
                self.multi_goal_prior_init.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_init,
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

        if self.goal_directed:
            self.multi_goal_prior_sample = []
            for i in range(self.num_goals):
                self.multi_goal_prior_sample.append(
                    UnaryFactor(
                        self.d_state_opt,
                        self.sigma_goal_sample,
                        self.multi_goal_states[i],
                        self.tensor_args,
                    )
                )

    def get_dist(
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

    def optimize(
            self,
            opt_iters=None,
            debug=False,
            **observation
    ):
        if opt_iters is None:
            opt_iters = self.opt_iters

        for opt_step in range(opt_iters):
            b, K = self._step(debug=debug, **observation)

            # stop criteria
            if self.stop_criteria is not None:
                costs = self._get_costs(b, K)
                if opt_step == 0:
                    costs_previous = costs.clone()
                    continue
                if torch.all(torch.abs((costs - costs_previous)/costs) < self.stop_criteria):
                    break
                costs_previous = costs.clone()

        self.costs = self._get_costs(b, K)

        position_seq_mean = self._particle_means[..., :self.n_dof].clone()
        velocity_seq_mean = self._particle_means[..., -self.n_dof:].clone()
        # costs = self.costs.clone()

        self._recent_state_trajectories = position_seq_mean
        self._recent_control_particles = velocity_seq_mean

        # get mean trajectory
        curr_traj = self._get_traj()
        return curr_traj

    def _step(self, debug=False, **observation):
        with TimerCUDA() as t_grad:
            A, b, K = self.cost.get_linear_system(
                self._particle_means,
                n_interpolated_points=self.n_interpolated_points,
                **observation)
        # if debug:
        #     print(f't_grad {t_grad}')

        J_t_J, g = self._get_grad_terms(
            A, b, K,
            delta=self.solver_params['delta'],
            trust_region=self.solver_params.get('trust_region', False),
            sparse_computation=self.solver_params.get('sparse_computation', False),
            sparse_computation_block_diag=self.solver_params.get('sparse_computation_block_diag', False)
        )

        with TimerCUDA() as t_solve:
            d_theta = self.get_torch_solve(
                J_t_J, g,
                method=self.solver_params['method'],
            )
        # if debug:
        #     print(f't_solve {t_solve}')

        d_theta = d_theta.view(
                self.num_particles,
                self.n_support_points,
                self.d_state_opt,
            )

        self._particle_means = self._particle_means + self.step_size * d_theta
        self._particle_means.detach_()

        return b, K

    def _get_grad_terms(
            self,
            A, b, K,
            delta=0.,
            trust_region=False,
            sparse_computation=False,
            sparse_computation_block_diag=False
    ):
        # Levenberg - Marquardt approximation

        B, M, N = A.shape
        if not sparse_computation:
            # Original implementation with dense matrices
            I = torch.eye(self.N, self.N, device=self.tensor_args['device'], dtype=A.dtype)
            A_t_K = A.transpose(-2, -1) @ K
            A_t_A = A_t_K @ A

            if not trust_region:
                J_t_J = A_t_A + delta * I
            else:
                # J_t_J = A_t_A + delta * I * torch.diagonal(A_t_A, dim1=-2, dim2=-1).unsqueeze(-1)
                # Since hessian will be averaged over particles, add diagonal matrix of the mean.
                diag_A_t_A = A_t_A.mean(0) * I
                J_t_J = A_t_A + delta * diag_A_t_A
            g = A_t_K @ b
        else:
            ################################################
            # Sparse matrix with loop over batch dimension
            if not sparse_computation_block_diag:
                I_sparse = torch.sparse_coo_tensor(
                    torch.arange(0, self.N).repeat(2, 1),
                    torch.ones(self.N),
                    (self.N, self.N),
                    device=self.tensor_args['device']
                )

                J_t_J_sparse_l = []
                g_sparse_l = []
                for A_, K_, b_ in zip(A, K, b):
                    A_t_K_sparse = torch.sparse.mm(A_.transpose(-2, -1).to_sparse(), K_.to_sparse())
                    A_t_A_sparse = torch.sparse.mm(A_t_K_sparse, A_.to_sparse())
                    if not trust_region:
                        J_t_J_sparse = A_t_A_sparse + delta * I_sparse
                    else:
                        diag_A_t_A_sparse = A_t_A_sparse.values().mean() * I_sparse
                        J_t_J_sparse = A_t_A_sparse + delta * diag_A_t_A_sparse

                    J_t_J_sparse_l.append(J_t_J_sparse)
                    g_sparse = torch.sparse.mm(A_t_K_sparse, b_.to_sparse())
                    g_sparse_l.append(g_sparse)

                J_t_J = torch.stack([x.to_dense() for x in J_t_J_sparse_l])
                g = torch.stack([x.to_dense() for x in g_sparse_l])
            else:
                ################################################
                # Sparse matrix with block diagonal, then extract block diagonals
                # TODO - ATTENTION!
                #  This implementation takes a lot of memory because of the block diagonal construction.
                I_sparse = torch.sparse_coo_tensor(
                    torch.arange(0, B * self.N).repeat(2, 1),
                    torch.ones(B * self.N),
                    (B * self.N, B * self.N),
                    device=self.tensor_args['device']
                )
                A_block_diag_sparse = torch.block_diag(*A).to_sparse()
                A_t_block_diag_sparse = torch.block_diag(*A.transpose(-2, -1)).to_sparse()
                K_block_diag_sparse = torch.block_diag(*K).to_sparse()

                A_t_K_sparse = torch.sparse.mm(A_t_block_diag_sparse, K_block_diag_sparse)
                A_t_A_sparse = torch.sparse.mm(A_t_K_sparse, A_block_diag_sparse)
                if not trust_region:
                    J_t_J_sparse = A_t_A_sparse + delta * I_sparse
                else:
                    raise NotImplementedError
                    # TODO - compute sparse version of diag_A_t_A_sparse
                    # diag_A_t_A_sparse = A_t_A.mean(0) * I_sparse
                    J_t_J_sparse = A_t_A_sparse + delta * diag_A_t_A_sparse

                J_t_J_block_diag = J_t_J_sparse.to_dense()
                g_sparse = torch.sparse.mm(A_t_K_sparse, torch.block_diag(*b).to_sparse())
                g_block_diag = g_sparse.to_dense()

                # TODO - remove for loop to extract block diagonals
                J_t_J = torch.stack([J_t_J_block_diag[i*N:(i+1)*N, i*N:(i+1)*N] for i in range(B)])
                g = torch.stack([g_block_diag[i * N:(i + 1) * N, i].unsqueeze(-1) for i in range(B)])

        return J_t_J, g

    def get_torch_solve(
        self,
        A, b,
        method,
    ):
        if method == 'inverse':
            res = torch.linalg.solve(A, b)
        elif method == 'cholesky':
            # method 1
            # old implementation - recheck torch.allclose(res, torch.linalg.solve(A, b))
            # l = torch.linalg.cholesky(A)
            # z = torch.linalg.solve_triangular(l, b, upper=False)
            # res = torch.linalg.solve_triangular(l.mT, z, upper=False)

            # method 2
            # z = torch.triangular_solve(b, l, transpose=False, upper=False)[0]
            # res = torch.triangular_solve(z, l, transpose=True, upper=False)[0]

            # method 3
            l, _ = torch.linalg.cholesky_ex(A)
            res = torch.cholesky_solve(b, l)

        elif method == 'cholesky-sparse':
            # TODO - ATTENTION!
            #  This implementation takes a lot of memory because of the block diagonal construction.
            raise NotImplementedError
            # if self.tensor_args['dtype'] == torch.float32:
            #     cholesky_fn = cholespy.CholeskySolverF
            # elif self.tensor_args['dtype'] == torch.float64:
            #     cholesky_fn = cholespy.CholeskySolverD
            # else:
            #     raise NotImplementedError

            # convert batch to block diagonal - https://github.com/rgl-epfl/cholespy/issues/26
            # TODO - creating a big block diagonal matrix leads to memory problems
            A_block_diag = torch.block_diag(*A)
            b_stacked = einops.rearrange(b, "b d 1 -> (b d) 1")

            A_sparse = A_block_diag.to_sparse(layout=torch.sparse_coo)

            cholesky_fn = cholespy.CholeskySolverF
            solver = cholesky_fn(
                A_sparse.size()[0],
                A_sparse.indices()[0], A_sparse.indices()[1], A_sparse.values(),
                cholespy.MatrixType.COO
            )
            res_ = torch.zeros_like(b_stacked)

            solver.solve(b_stacked, res_)

            # batchify results
            res = einops.rearrange(res_, "(b d) 1 -> b d 1", b=A.shape[0])

        elif method == 'lstq':
            # empirically slower
            res = torch.linalg.lstsq(A, b)[0]
        else:
            raise NotImplementedError

        return res

    def _get_costs(self, errors, w_mat):
        costs = errors.transpose(1, 2) @ w_mat.unsqueeze(0) @ errors
        return costs.reshape(self.num_particles,)

    def get_recent_samples(self):
        vel = self._recent_control_particles.detach().clone()
        vel = einops.rearrange(vel, '(m b) h d -> m b h d', m=self.num_goals)
        pos = self._recent_state_trajectories.detach().clone()
        pos = einops.rearrange(pos, '(m b) h d -> m b h d', m=self.num_goals)
        # pos_mean = self._particle_means[..., :self.n_dof].detach().clone()
        # vel_mean = self._particle_means[..., -self.n_dof:].detach().clone()

        return (
            pos,
            vel,
        )
