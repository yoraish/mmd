import numpy as np
import torch
from scipy import interpolate

from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy


def smoothen_trajectory(traj_pos, n_support_points=30, dt=0.02, set_average_velocity=True, zero_velocity=False,
                        tensor_args=None):
    """
    Smoothen a trajectory by interpolating it with a cubic spline and setting the average velocity.
    """
    assert not (set_average_velocity and zero_velocity), "Either sets the average velocity or zero velocity"
    traj_pos = to_numpy(traj_pos)
    try:
        # bc_type='clamped' for zero velocities at start and finish
        spline_pos = interpolate.make_interp_spline(np.linspace(0, 1, traj_pos.shape[0]), traj_pos, k=3,
                                                    bc_type='clamped')
        spline_vel = spline_pos.derivative(1)
    except:
        # Trajectory is too short to interpolate, so add last position again and interpolate
        traj_pos = np.vstack((traj_pos, traj_pos[-1] + np.random.normal(0, 0.01)))
        return smoothen_trajectory(traj_pos, n_support_points=n_support_points, dt=dt,
                                   set_average_velocity=set_average_velocity, zero_velocity=zero_velocity,
                                   tensor_args=tensor_args)

    pos = spline_pos(np.linspace(0, 1, n_support_points))
    vel = np.zeros_like(pos)
    if zero_velocity:
        pass
    elif set_average_velocity:
        avg_vel = (traj_pos[1] - traj_pos[0]) / (n_support_points * dt)
        vel[1:-1, :] = avg_vel
    else:
        vel = spline_vel(np.linspace(0, 1, n_support_points))

    return to_torch(pos, **tensor_args), to_torch(vel, **tensor_args)


def smoothen_trajectory_with_bc(traj_pos, grad_start, grad_goal, bc_weight=0.2, n_support_points=30, dt=0.02,
                                set_average_velocity=True, zero_velocity=False, tensor_args=None):
    traj_pos = to_numpy(traj_pos).T  # Shape is (q_pos_dim, H)
    adjusted_points = np.copy(traj_pos)
    if not grad_start is None and not grad_goal is None:
        grad_start = to_numpy(grad_start)
        grad_goal = to_numpy(grad_goal)
        print(traj_pos.shape, grad_start.shape, grad_goal.shape)
        # Adjust the control points based on derivative conditions
        adjusted_points[:, 0] += grad_start * bc_weight  # scale to adjust sensitivity
        adjusted_points[:, -1] -= grad_goal * bc_weight  # scale to adjust sensitivity
        # Add the original start and end points back in.
        adjusted_points = np.hstack((traj_pos[:, [0]], adjusted_points, traj_pos[:, [-1]]))

    # Create the spline
    tck, u = interpolate.splprep([adjusted_points[i] for i in range(len(adjusted_points))], s=0.1)

    # Evaluate the spline
    u_fine = np.linspace(0, 1, n_support_points)
    x_fine, y_fine = interpolate.splev(u_fine, tck)

    pos = np.stack((x_fine, y_fine), axis=0)
    vel = np.zeros_like(pos)
    if zero_velocity:
        pass
    elif set_average_velocity:
        avg_vel = (traj_pos[:, 1] - traj_pos[:, 0]) / (n_support_points * dt)
        vel[:, 1:-1] = avg_vel[:, None]
    else:
        vel = interpolate.splev(u_fine, tck, der=1)
    return to_torch(pos, **tensor_args).T, to_torch(vel, **tensor_args).T  # Shape is (H, q_pos_dim)


def interpolate_traj_via_points(trajs, num_interpolation=10):
    # Interpolates a trajectory linearly between waypoints
    H, D = trajs.shape[-2:]
    if num_interpolation > 0:
        assert trajs.ndim > 1
        traj_dim = trajs.shape
        alpha = torch.linspace(0, 1, num_interpolation + 2).type_as(trajs)[1:num_interpolation + 1]
        alpha = alpha.view((1,) * len(traj_dim[:-1]) + (-1, 1))
        interpolated_trajs = trajs[..., 0:traj_dim[-2] - 1, None, :] * alpha + \
                             trajs[..., 1:traj_dim[-2], None, :] * (1 - alpha)
        interpolated_trajs = interpolated_trajs.view(traj_dim[:-2] + (-1, D))
    else:
        interpolated_trajs = trajs
    return interpolated_trajs


def finite_difference_vector(x, dt=1., method='forward'):
    # finite differences with zero padding at the borders
    diff_vector = torch.zeros_like(x)
    if method == 'forward':
        diff_vector[..., :-1, :] = torch.diff(x, dim=-2) / dt
    elif method == 'backward':
        diff_vector[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / dt
    elif method == 'central':
        diff_vector[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2 * dt)
    else:
        raise NotImplementedError
    return diff_vector
