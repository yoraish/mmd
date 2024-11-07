import torch


def extend_path(distance_fn, q1, q2, max_step=0.03, max_dist=0.1, tensor_args=None):
    # max_dist must be <= radius of RRT star!
    dist = distance_fn(q1, q2)
    if dist > max_dist:
        q2 = q1 + (q2 - q1) * (max_dist / dist)

    alpha = torch.linspace(0, 1, int(dist / max_step) + 2).to(**tensor_args)  # skip first and last
    q1 = q1.unsqueeze(0)
    q2 = q2.unsqueeze(0)
    extension = q1 + (q2 - q1) * alpha.unsqueeze(1)
    return extension


def safe_path(sequence, collision_fn):
    in_collision = collision_fn(sequence)
    idxs_in_collision = torch.argwhere(in_collision)
    if idxs_in_collision.nelement() == 0:
        if sequence.ndim == 1:
            return sequence.reshape((1, -1))
        return sequence[-1].reshape((1, -1))
    else:
        first_idx_in_collision = idxs_in_collision[0]
        if first_idx_in_collision == 0:
            # the first point in the sequence is in collision
            return []
        # return the point immediate before the one in collision
        return sequence[first_idx_in_collision-1]


def purge_duplicates_from_traj(path, eps=1e-6):
    # Remove duplicated points from a trajectory
    if len(path) < 2:
        return path
    if isinstance(path, list):
        path = torch.stack(path, dim=0)
    if path.shape[0] == 2:
        return path

    abs_diff = torch.abs(torch.diff(path, dim=-2))
    row_idxs = torch.argwhere(torch.all((abs_diff > eps) == False, dim=-1) == False).unique()
    selection = path[row_idxs]
    # Always add the first and last elements of the path
    if torch.allclose(selection[0], path[0]) is False:
        selection = torch.cat((path[0].view(1, -1), selection), dim=0)
    if torch.allclose(selection[-1], path[-1]) is False:
        selection = torch.cat((selection, path[-1].view(1, -1)), dim=0)
    return selection
