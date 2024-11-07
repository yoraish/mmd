import numpy as np
import torch

from torch_kinematics_tree.geometrics.manifold import Manifold


def compute_traj_derivatives(traj, dt, manifold=None, smooth=False):
    """
    Estimate the trajectory dynamics.
    Parameters
    ----------
    :param traj (torch of shape (length, dim_M)): trajectory.
    :param smooth (boolean): whether smoothing is applied to traj first.
    Returns
    ----------
    :return traj (torch of shape (length, dim_M)): pose trajectory without/with smoothing.
    :return d_traj (torch of shape (length, dim_T)): first derivative of traj.
    :return dd_traj (torch of shape (length, dim_T)): second derivative of traj.
    """
    dim_M = traj.shape[-1]
    if not manifold:
        # if no manifold specified, Euclidean manifold is used.
        manifold = Manifold.get_euclidean_manifold(dim_M)
    # smoothing first
    if smooth:
        traj = smooth_traj(traj, manifold=manifold)
    # compute derivatives
    d_traj = compute_traj_velocity(traj, dt, manifold=manifold)
    dd_traj = compute_traj_velocity(d_traj, dt)
    # copy the last 2 entries (corner case)
    dd_traj[-2, :] = dd_traj[-3, :]
    dd_traj[-1, :] = dd_traj[-3, :]
    return traj, d_traj, dd_traj


def smooth_traj(traj, manifold=None, window_length=30, beta=14):
    """
    Smooth the pose using the kaiser window np.kaiser(window_length, beta).
    Parameters
    ----------
    :param traj (torch of shape (length, dim_M)): trajectory.
    :param window_length (int): the length of the window. (Default: 30).
    :param beta (float): shape parameter for the kaiser window. (Default: 14).
    Returns
    ----------
    :return smooth_traj (torch of shape (length, dim_M)): pose trajectory after smoothing.
    """
    length, dim_M = traj.shape[-2:]
    tensor_args = {'device': traj.device, 'dtype': traj.dtype}
    if not manifold:
        manifold = Manifold.get_euclidean_manifold(dim_M)
    if dim_M != manifold.dim_M:
        raise ValueError('[Trajectory]: Input X shape %s is not consistent with manifold.dim_M %s' % (dim_M, manifold.dim_M))
    # apply kaiser filter
    half_wl = int(window_length / 2)
    window = torch.kaiser_window(window_length, periodic=False, beta=beta, **tensor_args).unsqueeze(-1)
    smooth_traj = traj.detach().clone()
    for t in range(length):
        weights = window[max(half_wl - t, 0):(2 * half_wl - max(0, t + half_wl - length))]
        smooth_traj[t, :] = manifold.mean(smooth_traj[max(t - half_wl, 0):(t + half_wl), :], weights=weights)
    return smooth_traj


def compute_traj_velocity(traj, dt, manifold=None):
    """
    Estimate the first derivative of input trajectory traj.
    Parameters
    ----------
    :param traj (torch of shape (length, dim_M)): trajectory.
    :param dt (float): sampling time associated with traj.
    Returns
    ----------
    :return d_traj (torch of shape (length, dim_T)): first derivative of traj.
    """
    length, dim_M = traj.shape[-2:]
    if not manifold:
        manifold = Manifold.get_euclidean_manifold(dim_M)
    if dim_M != manifold.dim_M:
        raise ValueError('[Trajectory]: Input X shape %s is not consistent with manifold.dim_M %s' % (dim_M, manifold.dim_M))
    # estimate d_traj
    d_traj = []
    for t in range(length - 1):
        d_traj_t = manifold.log_map(traj[t + 1, :], base=traj[t, :]) / dt
        d_traj.append(d_traj_t.squeeze())
    d_traj.append(d_traj_t.squeeze())
    d_traj = torch.stack(d_traj, dim=0)
    names = np.array(manifold.name.split(' x '))
    if 'S^3' in names:
        indices = np.where(names == 'S^3')[0]
        for i in indices:
            d_traj[:, i * 3:i * 3 + 3] *= 2  # convert to angular
    if d_traj.shape[-1] != manifold.dim_T:
        raise ValueError('[Trajectory]: d_traj shape %s is not consistent with manifold.dim_T %s' % (d_traj.shape[0], manifold.dim_T))
    if d_traj.shape[-2] != length:
        raise ValueError('[Trajectory]: Length of d_traj %s is not consistent with input traj length %s' % (d_traj.shape[1], length))
    return d_traj


def interpolate(p1, p2, d_param, dist):
    """ interpolate between points """
    alpha = min(max(d_param / dist, 0.), 1.)
    return (1. - alpha) * p1 + alpha * p2
