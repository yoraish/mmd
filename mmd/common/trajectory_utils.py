"""
MIT License

Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.
import torch
from scipy.signal import savgol_filter
# Project imports.
from mmd.config.mmd_params import MMDParams as params


def smooth_trajs(trajs, window_size=10, poly_order=2):
    """
    Smooth the trajectories.
    trajs: List of trajectories. Each is a tensor of shape (H, q_dim).
    """
    if isinstance(trajs, torch.Tensor):
        assert trajs.dim() == 3  # (B, H, q_dim)
        smoothed_trajs = savgol_filter(trajs.cpu().numpy(), window_size, poly_order, axis=1)
        return torch.tensor(smoothed_trajs).to(trajs.device)

    smoothed_trajs = []
    for traj in trajs:
        traj = traj.cpu().numpy()
        window_size_traj = min(window_size, traj.shape[0])
        if window_size_traj <= 2:
            smoothed_trajs.append(torch.tensor(traj).to(**params.tensor_args))
            continue
        smoothed_traj = savgol_filter(traj, window_size_traj, poly_order, axis=0)
        smoothed_traj = torch.tensor(smoothed_traj).to('cuda')
        smoothed_trajs.append(smoothed_traj)
    return smoothed_trajs


def densify_trajs(trajs, n_points_interp=10):
    """
    Densify the trajectories.
    :param trajs: List of trajectories. Each is a tensor of shape (H, n_dims).
    :param n_points_interp: Number of points to interpolate between each pair of points.
    """
    densified_trajs = []
    for traj in trajs:
        densified_traj = []
        for i in range(traj.shape[0] - 1):
            densified_traj.append(traj[i])
            for j in range(1, n_points_interp):
                densified_traj.append(traj[i] + j * (traj[i + 1] - traj[i]) / n_points_interp)
        densified_traj.append(traj[-1])
        densified_traj = torch.stack(densified_traj)
        densified_trajs.append(densified_traj)
    return densified_trajs


def are_points_closer_than_margin(points_batch, margin):
    """
    Check proximity between points. (Sometimes this is for Robot-robot collisions).
    Args:
        points_batch: (..., n_points, q_dim)
    Returns:
        collisions: (..., n_points, n_points), True if there is the pair of points is closer than the margin.
    """
    # Check collisions between robots.
    assert points_batch.dim() >= 2
    points_batch1 = points_batch.unsqueeze(-2)
    points_batch2 = points_batch.unsqueeze(-3)
    # Pairwise distances between robots.
    robot_dq = points_batch1 - points_batch2
    # (..., n_robots, n_robots)
    robot_dq_norm = torch.norm(robot_dq, dim=-1)
    # (..., n_robots, n_robots)
    collisions = robot_dq_norm < margin
    # Set the trace to be False.
    collisions = collisions & ~torch.eye(collisions.shape[-1], device=collisions.device, dtype=collisions.dtype)
    return collisions
