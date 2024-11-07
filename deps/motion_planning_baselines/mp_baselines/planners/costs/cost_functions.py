from abc import ABC, abstractmethod

import einops
import numpy as np
import torch
from typing import List, Optional

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.costs.factors.field_factor import FieldFactor
from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_planning_objectives.fields.distance_fields import interpolate_points_v1
from torch_robotics.torch_utils.torch_utils import batched_weighted_dot_prod
from torch_robotics.trajectory.utils import finite_difference_vector


class Cost(ABC):
    def __init__(self, robot, n_support_points, tensor_args=None, **kwargs):
        self.robot = robot
        self.n_dof = robot.q_dim
        self.dim = 2 * self.n_dof  # position + velocity
        self.n_support_points = n_support_points

        self.tensor_args = tensor_args

    def set_cost_factors(self):
        pass

    def __call__(self, trajs, **kwargs):
        return self.eval(trajs, **kwargs)

    @abstractmethod
    def eval(self, trajs, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajs, **kwargs):
        pass

    def get_q_pos_vel_and_fk_map(self, trajs, **kwargs):
        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:
            N, B, H, D = trajs.shape  # n_goals (or steps), batch of trajectories, length, dim
            trajs = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape

        q_pos = self.robot.get_position(trajs)
        q_vel = self.robot.get_velocity(trajs)
        H_positions = self.robot.fk_map_collision(
            q_pos)  # I, taskspaces, x_dim+1, x_dim+1 (homogeneous transformation matrices)
        return trajs, q_pos, q_vel, H_positions


class CostComposite(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            cost_list,
            weights_cost_l=None,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.cost_l = cost_list
        self.weight_cost_l = weights_cost_l if weights_cost_l is not None else [1.0] * len(cost_list)

    def eval(self, trajs, trajs_interpolated=None, return_invidual_costs_and_weights=False, **kwargs):
        trajs, q_pos, q_vel, H_positions = self.get_q_pos_vel_and_fk_map(trajs)

        if not return_invidual_costs_and_weights:
            cost_total = 0
            for cost, weight_cost in zip(self.cost_l, self.weight_cost_l):
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    if isinstance(cost, CostCollision):
                        trajs_tmp = trajs_interpolated
                    else:
                        trajs_tmp = trajs
                else:
                    trajs_tmp = trajs
                cost_tmp = weight_cost * cost(trajs_tmp, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions, **kwargs)
                cost_total += cost_tmp
            return cost_total
        else:
            cost_l = []
            for cost in self.cost_l:
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    if isinstance(cost, CostCollision):
                        trajs_tmp = trajs_interpolated
                    else:
                        trajs_tmp = trajs
                else:
                    trajs_tmp = trajs

                cost_tmp = cost(trajs_tmp, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions, **kwargs)
                cost_l.append(cost_tmp)

            if return_invidual_costs_and_weights:
                return cost_l, self.weight_cost_l

    def get_linear_system(self, trajs, n_interpolated_points=None, **kwargs):
        trajs.requires_grad = True
        # TODO - join fk map into one call for trajs and trajs_interp
        trajs, q_pos, q_vel, H_positions = self.get_q_pos_vel_and_fk_map(trajs)

        # Upsample trajectory for finer collision checking
        # Interpolate in joint space
        # TODO - change from linear interpolation to GP interpolation
        if n_interpolated_points is None:
            trajs_interp, q_pos_interp, q_vel_interp, H_positions_interp = None, None, None, None
        else:
            trajs_interp = interpolate_points_v1(trajs, n_interpolated_points)
            trajs_interp, q_pos_interp, q_vel_interp, H_positions_interp = self.get_q_pos_vel_and_fk_map(trajs_interp)

        batch_size = trajs.shape[0]
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost, weight_cost in zip(self.cost_l, self.weight_cost_l):
            A, b, K = cost.get_linear_system(
                trajs, q_pos=q_pos, q_vel=q_vel, H_positions=H_positions,
                trajs_interp=trajs_interp, q_pos_interp=q_pos_interp, q_vel_interp=q_vel_interp,
                H_positions_interp=H_positions_interp,
                **kwargs)
            if A is None or b is None or K is None:
                continue
            optim_dim += A.shape[1]
            As.append(A.detach())
            bs.append(b.detach())
            Ks.append(K.detach())

        A = torch.cat(As, dim=1)
        b = torch.cat(bs, dim=1)
        K = torch.zeros(batch_size, optim_dim, optim_dim, **self.tensor_args)
        offset = 0
        for i in range(len(Ks)):
            dim = Ks[i].shape[1]
            K[:, offset:offset + dim, offset:offset + dim] = Ks[i]
            offset += dim
        return A, b, K


class CostCollision(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            field=None,
            sigma_coll=None,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.obst_factor = None
        self.field = field
        self.sigma_coll = sigma_coll

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.obst_factor = FieldFactor(
            self.n_dof,
            self.sigma_coll,
            # [45, 50]  # Can consider also only a subset of the trajectory.
            [1, None]  # Take the whole trajectory except for the first point.
        )

    def eval(self, trajs, q_pos=None, q_vel=None, H_positions=None, **observation):
        costs = 0
        if self.field is not None:
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            H_pos = H_positions
            err_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                q_pos=q_pos,
                q_vel=q_vel,
                H_pos=H_pos,
                calc_jacobian=False,  # TODO: NOTE(an): no need for grads in StochGPMP
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )
            w_mat = self.obst_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(self, trajs, q_pos=None, q_vel=None, H_positions=None,
                          trajs_interp=None, q_pos_interp=None, q_vel_interp=None, H_positions_interp=None,
                          **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            # H_pos = link_pos_from_link_tensor(H)  # get translation part from transformation matrices
            H_pos = H_positions

            # Get H_obst wrt to the interpolated trajectory. This is computed inside get_error

            err_obst, H_obst = self.obst_factor.get_error(
                trajs,
                self.field,
                q_pos=q_pos,
                q_vel=q_vel,
                H_pos=H_pos,
                trajs_interp=trajs_interp,
                q_pos_interp=q_pos_interp,
                q_vel_interp=q_vel_interp,
                H_pos_interp=H_positions_interp,
                calc_jacobian=True,
                obstacle_spheres=observation.get('obstacle_spheres', None)
            )

            A = torch.zeros(batch_size, self.n_support_points - 1, self.dim * self.n_support_points, **self.tensor_args)
            A[:, :, :H_obst.shape[-1]] = H_obst
            # shift each row by self.dim
            idxs = torch.arange(A.shape[-1], **self.tensor_args).repeat(A.shape[-2], 1)
            idxs = (idxs - torch.arange(self.dim, (idxs.shape[0] + 1) * self.dim, self.dim, **self.tensor_args).view(-1, 1)) % idxs.shape[-1]
            idxs = idxs.to(torch.int64)
            A = torch.gather(A, -1, idxs.repeat(batch_size, 1, 1))

            # old code not vectorized
            # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L275

            b = err_obst.unsqueeze(-1)
            K = self.obst_factor.K * torch.eye((self.n_support_points - 1), **self.tensor_args).repeat(batch_size, 1, 1)

        return A, b, K


# class CostConstraint(Cost):
#     """
#     Currently implementing a "vertex constraint."
#     Avoid this configuration at this time (interval).
#     """
#     def __init__(
#             self,
#             robot,
#             n_support_points,
#             q: torch.Tensor,  # Shape: (n_dof,)
#             traj_range: tuple,
#             radius: float,
#             sigma=None,
#             **kwargs
#     ):
#         super().__init__(robot, n_support_points, **kwargs)
#         self.q = q.to(**self.tensor_args)
#         self.traj_range = traj_range
#         self.radius = radius
#         self.sigma = sigma
#
#     def eval(self, trajs, **observation):
#         costs = 0
#         if self.q is not None:
#             q_pos = self.robot.get_position(trajs)
#             q_pos = q_pos[:, self.traj_range[0]:self.traj_range[1], :]
#             dist_constraint = torch.norm(q_pos - self.q, dim=-1)
#             mask = dist_constraint > self.radius
#             dist_constraint = torch.where(mask, torch.zeros_like(dist_constraint), dist_constraint)
#             costs = self.radius - dist_constraint
#             costs = costs.sum(-1)
#         return costs
#
#     def get_linear_system(self, trajs, **kwargs):
#         pass


class CostConstraint(Cost):
    """
    Implements multiple "vertex constraints" in a fully vectorized manner.
    Each constraint is associated with a specific configuration and time interval.
    """

    def __init__(
            self,
            robot,
            n_support_points,
            q_l: List[torch.Tensor],  # Length n.
            traj_range_l: List[tuple],  # Length n. A tuple (start, end) for each constraint. Inclusive.
            radius_l: List[float],  # Length n. The radius for each constraint.
            is_soft: bool = False,  # Sigma stands for the standard deviation of the cost.
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.qs = torch.stack([q.to(**self.tensor_args) for q in q_l], dim=0)  # (n, q_dim)
        self.traj_ranges = torch.tensor(traj_range_l, **self.tensor_args)  # (n, 2)
        self.radii = torch.tensor(radius_l, device=self.qs.device)  # (n,)
        self.is_soft = is_soft

    def eval(self, trajs, **observation):
        q_pos = self.robot.get_position(trajs)  # (B, H, q_dim)
        # Get the start and end indices for each range.
        start_indices = self.traj_ranges[:, 0]  # (n,)
        end_indices = self.traj_ranges[:, 1]  # (n,)

        # Prepare a mask to extract relevant positions in the trajectory for each constraint.
        mask = torch.arange(q_pos.shape[1], device=q_pos.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        mask = (mask >= start_indices.view(-1, 1, 1)) & (mask < end_indices.view(-1, 1, 1))  # (n, 1, H)

        # Repeat and reshape q_pos to match the constraints dimensions.
        q_pos = q_pos.unsqueeze(0).expand(self.qs.shape[0], -1, -1, -1)  # (n, B, H, q_dim)
        q_pos_masked = q_pos * mask.unsqueeze(-1)  # (n, B, H, q_dim)

        # Compute the distance between each position in the range and the corresponding q.
        dist_constraint = torch.norm(q_pos_masked - self.qs.view(-1, 1, 1, q_pos.shape[-1]),
                                     dim=-1)  # (n, B, H)

        # Apply the radius constraints.
        mask_radius = dist_constraint > self.radii.view(-1, 1, 1)  # (n, B, H)
        dist_constraint = torch.where(mask_radius, torch.zeros_like(dist_constraint),
                                      dist_constraint)  # Only penalize inside the radius

        # Compute the cost.
        costs = (self.radii.view(-1, 1, 1) - dist_constraint).sum(dim=-1)  # (n, B)

        # Sum the costs across all constraints.
        total_costs = costs.sum()  # (1,)

        return total_costs

    def get_linear_system(self, trajs, **kwargs):
        pass


class CostMaxVelocity(Cost):
    def __init__(
            self,
            robot,
            n_support_points,
            max_vel=None,
            sigma_max_vel=None,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.max_vel = max_vel
        self.sigma_max_vel = sigma_max_vel

    def eval(self, trajs, **observation):
        costs = 0
        q_pos_trajs = self.robot.get_position(trajs)
        if self.max_vel is not None:
            vel = finite_difference_vector(q_pos_trajs, dt=self.robot.dt, method='central')
            cost = torch.pow(vel - self.max_vel, 2)
            costs = cost.sum(-1)
        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostVelocityAndDirectionAlignment(Cost):
    def __init__(
            self,
            robot,
            n_support_points,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)

    def eval(self, trajs, **observation):
        q_direction_trajs = self.robot.get_direction_vector(trajs)
        q_spatial = self.robot.get_spatial_component(trajs)
        q_spatial_vel = finite_difference_vector(q_spatial, dt=self.robot.dt, method='central')
        q_spatial_vel = q_spatial_vel / torch.norm(q_spatial_vel, dim=-1, keepdim=True)
        cost = torch.sum(q_direction_trajs * q_spatial_vel, dim=-1)  # Just dot.
        costs = cost.sum(-1)
        return costs

    def get_linear_system(self, trajs, **observation):
        batch_size = trajs.shape[0]
        dim = trajs.shape[-1]

        # Initialize the linear system components A, b, and K.
        A = torch.zeros(batch_size, dim * self.n_support_points, dim * self.n_support_points, **self.tensor_args)
        b = torch.zeros(batch_size, dim * self.n_support_points, 1, **self.tensor_args)
        K = torch.zeros(batch_size, dim * self.n_support_points, dim * self.n_support_points, **self.tensor_args)

        # Get direction and velocity trajectories
        q_direction_trajs = self.robot.get_direction_vector(trajs)
        q_spatial = self.robot.get_spatial_component(trajs)
        q_spatial_vel = finite_difference_vector(q_spatial, dt=self.robot.dt, method='central')
        q_spatial_vel = q_spatial_vel / torch.norm(q_spatial_vel, dim=-1, keepdim=True)

        # Calculate the error as the difference between direction and normalized velocity vectors
        error = q_direction_trajs - q_spatial_vel

        # Now, populate A, b, and K matrices.
        # A will represent the linearized relationship between the deviation and the state
        # b will represent the error between the direction and velocity
        # K will represent a weighting matrix for this cost (often identity or diagonal)

        # For simplicity, we assume a diagonal weighting matrix (you can adjust based on problem setup).
        K_weight = torch.eye(dim, **self.tensor_args)
        K_weight = K_weight.unsqueeze(0).repeat(batch_size, self.n_support_points, 1, 1)

        # Fill in the matrices with appropriate dimensions
        for i in range(self.n_support_points):
            idx_start = i * dim
            idx_end = (i + 1) * dim

            # Populate A matrix for this support point (Jacobian of the error wrt the state)
            A[:, idx_start:idx_end, idx_start:idx_end] = K_weight[:, i]

            # Populate b with the error at this support point
            b[:, idx_start:idx_end] = error[:, i].unsqueeze(-1)

            # Populate K with the weighting matrix
            K[:, idx_start:idx_end, idx_start:idx_end] = K_weight[:, i]

        return A, b, K


class CostGP(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            start_state,
            dt,
            sigma_params,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.start_state = start_state
        self.dt = dt

        self.sigma_start = sigma_params['sigma_start']
        self.sigma_gp = sigma_params['sigma_gp']

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.dim,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)
        # Start cost
        err_p = self.start_prior.get_error(trajs[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP Trajectory cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        return costs

    def get_linear_system(self, trajs, **observation):
        batch_size = trajs.shape[0]
        A = torch.zeros(batch_size, self.dim * self.n_support_points, self.dim * self.n_support_points,
                        **self.tensor_args)
        b = torch.zeros(batch_size, self.dim * self.n_support_points, 1, **self.tensor_args)
        K = torch.zeros(batch_size, self.dim * self.n_support_points, self.dim * self.n_support_points,
                        **self.tensor_args)

        # Start prior factor
        err_p, H_p = self.start_prior.get_error(trajs[:, [0]])
        A[:, :self.dim, :self.dim] = H_p
        b[:, :self.dim] = err_p
        K[:, :self.dim, :self.dim] = self.start_prior.K

        # GP factors
        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajs)

        A[:, self.dim:, :-self.dim] = torch.block_diag(*H1_gp)
        A[:, self.dim:, self.dim:] += torch.block_diag(*H2_gp)
        b[:, self.dim:] = einops.rearrange(err_gp, "b h d 1 -> b (h d) 1")
        K[:, self.dim:, self.dim:] += torch.block_diag(*self.gp_prior.Q_inv)

        # old code not vectorized
        # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L161

        return A, b, K


class CostGPTrajectory(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            dt,
            sigma_gp=None,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.dt = dt

        self.sigma_gp = sigma_gp

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs, **observation):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostGPTrajectoryPositionOnlyWrapper(CostGPTrajectory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, trajs, **observation):
        vel = finite_difference_vector(trajs, dt=self.dt, method='central')
        trajs_tmp = torch.cat((trajs, vel), dim=-1)
        return super().eval(trajs_tmp, **observation)


class CostSmoothnessCHOMP(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.dt = robot.dt

        self.Sigma_inv = CHOMP._get_R_mat(dt=self.dt, n_support_points=n_support_points, **kwargs)

    def eval(self, trajs, **observation):
        R_mat = self.Sigma_inv
        cost = batched_weighted_dot_prod(trajs, R_mat, trajs)
        return cost

    def get_linear_system(self, trajs, **observation):
        pass


class CostJointLimits(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            eps=np.deg2rad(3),
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)

        self.eps = eps

    def eval(self, trajs, **observation):
        assert trajs.ndim == 3

        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)
        trajs_pos = self.robot.get_position(trajs)

        idxs_lower = torch.argwhere(trajs_pos < self.robot.q_min + self.eps)
        cost_lower = torch.pow(
            self.robot.q_min[idxs_lower[:, 2]] + self.eps - trajs_pos[
                idxs_lower[:, 0], idxs_lower[:, 1], idxs_lower[:, 2]],
            2
        ).sum(-1)

        idxs_upper = torch.argwhere(trajs_pos > self.robot.q_max - self.eps)
        cost_upper = torch.pow(
            self.robot.q_max[idxs_upper[:, 2]] - self.eps - trajs_pos[
                idxs_upper[:, 0], idxs_upper[:, 1], idxs_upper[:, 2]],
            2
        ).sum(-1)

        costs = (cost_lower + cost_upper)

        return costs

    def get_linear_system(self, trajs, **observation):
        pass


class CostGoal(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            field=None,
            sigma_goal=None,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.field = field
        self.sigma_goal = sigma_goal

        self.set_cost_factors()

    def set_cost_factors(self):
        #========= Cost factors ===============
        self.goal_factor = FieldFactor(
            self.n_dof,
            self.sigma_goal,
            [-1, None]  # only take last point
        )

    def eval(self, trajs, x_trajs=None, **observation):
        costs = 0
        if self.field is not None:
            err_obst = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=False,  # NOTE(an): no need for grads in StochGPMP
            )
            w_mat = self.goal_factor.K
            obst_costs = w_mat * err_obst.sum(1)
            costs = obst_costs

        return costs

    def get_linear_system(self, trajs, x_trajs=None, **observation):
        A, b, K = None, None, None
        if self.field is not None:
            batch_size = trajs.shape[0]
            A = torch.zeros(batch_size, 1, self.dim * self.n_support_points, **self.tensor_args)
            err_goal, H_goal = self.goal_factor.get_error(
                trajs,
                self.field,
                x_trajs=x_trajs,
                calc_jacobian=True,
            )
            A[:, :, -self.dim:(-self.dim + self.n_dof)] = H_goal
            b = err_goal.unsqueeze(-1)
            K = self.goal_factor.K * torch.eye(1, **self.tensor_args).repeat(batch_size, 1, 1)
        return A, b, K


class CostGoalPrior(Cost):

    def __init__(
            self,
            robot,
            n_support_points,
            multi_goal_states=None,  # num_goal x n_dim (pos + vel)
            num_particles_per_goal=None,
            num_samples=None,
            sigma_goal_prior=None,
            **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.multi_goal_states = multi_goal_states
        self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.sigma_goal_prior = sigma_goal_prior

        self.set_cost_factors()

    def set_cost_factors(self):
        self.multi_goal_prior = []
        # TODO: remove this for loop
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.dim,
                    self.sigma_goal_prior,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def eval(self, trajs, **observation):
        costs = 0
        if self.multi_goal_states is not None:
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal * self.num_samples, self.n_support_points,
                              self.dim)
            costs = torch.zeros(self.num_goals, self.num_particles_per_goal * self.num_samples, **self.tensor_args)
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]], calc_jacobian=False)
                w_mat = self.multi_goal_prior[i].K
                goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
                goal_costs = goal_costs.squeeze()
                costs[i] += goal_costs
            costs = costs.flatten()
        return costs

    def get_linear_system(self, trajs, **observation):
        A, b, K = None, None, None
        if self.multi_goal_states is not None:
            npg = self.num_particles_per_goal
            batch_size = npg * self.num_goals
            x = trajs.reshape(self.num_goals, self.num_particles_per_goal, self.n_support_points, self.dim)
            A = torch.zeros(batch_size, self.dim, self.dim * self.n_support_points, **self.tensor_args)
            b = torch.zeros(batch_size, self.dim, 1, **self.tensor_args)
            K = torch.zeros(batch_size, self.dim, self.dim, **self.tensor_args)
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g, H_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]])
                A[i * npg: (i + 1) * npg, :, -self.dim:] = H_g
                b[i * npg: (i + 1) * npg] = err_g
                K[i * npg: (i + 1) * npg] = self.multi_goal_prior[i].K

        return A, b, K
