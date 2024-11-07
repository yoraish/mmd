import torch

"""
This file implements the FieldFactor class, which is used to compute the error of a trajectory with respect to a field.

"""

class FieldFactor:

    def __init__(
            self,
            n_dof,
            sigma,
            traj_range,
    ):
        self.sigma = sigma
        self.n_dof = n_dof
        self.traj_range = traj_range
        self.K = 1. / (sigma**2)

    """
    Compute the error of the trajectory with respect to the field. The error is the negative log likelihood of the trajectory
    """
    def get_error(
            self,
            q_trajs,
            field,
            q_pos=None,
            q_vel=None,
            H_pos=None,
            q_trajs_interp=None,
            q_pos_interp=None,
            q_vel_interp=None,
            H_pos_interp=None,
            calc_jacobian=True,
            **kwargs
    ):
        batch = q_trajs.shape[0]

        if H_pos is not None:
            states = H_pos[:, self.traj_range[0]:self.traj_range[1]]
        else:
            states = q_trajs[:, self.traj_range[0]:self.traj_range[1], :self.n_dof].reshape(-1, self.n_dof)
        q_pos_new = q_pos[:, self.traj_range[0]:self.traj_range[1], :]
        length = q_pos_new.shape[-2]
        error = field.compute_cost(q_pos_new, states, **kwargs).reshape(batch, length)
        # Clip out all errors that are smaller than -0.02.
        error = torch.clamp(error, min=-0.02)

        if calc_jacobian:
            # compute jacobian wrt to the error of the interpolated trajectory
            error_interp = error
            if H_pos_interp is not None or q_trajs_interp is not None:
                # interpolated trajectory
                if H_pos_interp is not None:
                    states = H_pos_interp[:, self.traj_range[0]:self.traj_range[1]]
                else:
                    states = q_trajs_interp[:, self.traj_range[0]:self.traj_range[1], :self.n_dof].reshape(-1, self.n_dof)
                q_pos_new = q_pos_interp[:, self.traj_range[0]:self.traj_range[1], :]
                length = q_pos_new.shape[-2]
                error_interp = field.compute_cost(q_pos_new, states, **kwargs).reshape(batch, length)

            H = -torch.autograd.grad(error_interp.sum(), q_trajs, retain_graph=True)[0][:, self.traj_range[0]:self.traj_range[1], :self.n_dof]
            error = error.detach()
            field.zero_grad()
            return error, H
        else:
            return error
