import numpy as np
import torch
import torch.nn.functional as F


'''
"Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018. TABLE I
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
Format: q = [w, x, y, z]. # NOTE: change to [x, y, z, w] for pyBullet
'''


def sqrt_with_mask(x):
    """
    torch.sqrt(torch.max(0, x))
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def q_exp_map(v, base=None):
    v = v.unsqueeze(0) if len(v.shape) == 1 else v
    tensor_args = {'device': v.device, 'dtype': v.dtype}
    batch_dim = v.shape[:-1]
    if base is None:
        norm_v = torch.norm(v, dim=-1)
        q = torch.zeros(batch_dim + (4,), **tensor_args)
        q[..., 0] = 1.
        non_0 = torch.where(norm_v > 0)
        non_0_norm = norm_v[non_0].unsqueeze(-1)
        q[non_0] = torch.cat((
            torch.cos(non_0_norm),
            (torch.sin(non_0_norm) / non_0_norm) * v[non_0]), dim=-1)
        return q
    else:
        return q_mul(base, q_exp_map(v))


def q_log_map(q, base=None, eps=1e-10):
    q = q.unsqueeze(0) if len(q.shape) == 1 else q
    tensor_args = {'device': q.device, 'dtype': q.dtype}
    batch_dim = q.shape[:-1]
    if base is None:
        norm_q = torch.norm(q[..., 1:], dim=-1)
        non_0 = torch.where((norm_q > 0) * (torch.abs(q[..., 0]) <= 1))  # eps for numerical stability
        q_non_singular = q[non_0]
        non_0_norm = norm_q[non_0].unsqueeze(-1)
        acos = torch.acos(q_non_singular[..., 0]).unsqueeze(-1)
        acos[torch.where(q_non_singular[..., 0] < 0)] += -np.pi  # q and -q maps to the same point in SO(3)
        v = torch.zeros(batch_dim + (3,), **tensor_args)
        # print(q_non_singular[..., 1:].shape, (acos / non_0_norm.unsqueeze(-1)).repeat((1,)*len(batch_dim) + (3,)).shape)
        v[non_0] = q_non_singular[..., 1:] * (acos / non_0_norm)
        return v
    else:
        return q_log_map(q_mul(q_inverse(base), q))


def q_parallel_transport(p_g, g, h, eps=1e-10):
    p_g = p_g.unsqueeze(0) if len(p_g.shape) == 1 else p_g
    g, h = g.squeeze(), h.squeeze()
    tensor_args = {'device': p_g.device, 'dtype': p_g.dtype}
    Q_g = q_to_quaternion_matrix(g).squeeze()
    Q_h = q_to_quaternion_matrix(h).squeeze()
    B = torch.cat([
        torch.zeros((1, 3)), torch.eye(3)
    ], dim=0).to(**tensor_args)
    log_g_h = q_log_map(h, base=g)
    m = torch.norm(log_g_h, dim=-1)
    if m < eps:  # divide by zero
        return p_g
    q_temp = torch.zeros((1, 4), **tensor_args)
    q_temp[0, 1:] = log_g_h / m
    u = (Q_g @ q_temp.unsqueeze(-1)).squeeze()
    I4 = torch.eye(4).to(**tensor_args)
    R_g_h = I4 - torch.sin(m) * torch.outer(g, u) + (torch.cos(m) - 1) * torch.outer(u, u)
    A_g_h = B.T @ Q_h.T @ R_g_h @ Q_g @ B
    res = (A_g_h @ p_g.unsqueeze(-1)).squeeze(-1)
    return res


def q_mul(q1, q2):
    res = q_to_quaternion_matrix(q1) @ q2.unsqueeze(-1)
    return res.squeeze(-1)


def q_inverse(q):
    tensor_args = {'device': q.device, 'dtype': q.dtype}
    scaling = torch.tensor([1, -1, -1, -1], **tensor_args)
    return q * scaling / q_norm_squared(q)


def q_div(q1, q2):
    return q_mul(q1, q_inverse(q2))


def q_norm_squared(q):
    return torch.sum(q ** 2, dim=-1, keepdim=True)


def q_to_rotation_matrix(q):
    q = q.unsqueeze(0) if len(q.shape) == 1 else q
    w, x, y, z = torch.unbind(q, -1)
    double_cover = 2.0 / (q ** 2).sum(-1)
    o = torch.stack(
        (
            1 - double_cover * (y * y + z * z),
            double_cover * (x * y - z * w),
            double_cover * (x * z + y * w),
            double_cover * (x * y + z * w),
            1 - double_cover * (x * x + z * z),
            double_cover * (y * z - x * w),
            double_cover * (x * z - y * w),
            double_cover * (y * z + x * w),
            1 - double_cover * (x * x + y * y),
        ),
        dim=-1
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def q_to_quaternion_matrix(q):
    q = q.unsqueeze(0) if len(q.shape) == 1 else q
    w, x, y, z = torch.unbind(q, -1)
    o = torch.stack([
        w, -x, -y, -z,
        x, w, -z, y,
        y, z, w, -x,
        z, -y, x, w
    ], dim=-1)
    return o.reshape(q.shape[:-1] + (4, 4))


def rotation_matrix_to_q(rot_mat):
    rot_mat = rot_mat.unsqueeze(0) if len(rot_mat.shape) == 1 else rot_mat
    assert rot_mat.size(-1) == 3 or rot_mat.size(-2) == 3
    batch_dim = rot_mat.shape[:-2]
    tensor_args = {'device': rot_mat.device, 'dtype': rot_mat.dtype}
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot_mat.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = sqrt_with_mask(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    ).to(**tensor_args)
    quat_by_wxyz = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    ).to(**tensor_args)
    flr = torch.tensor(0.1).to(**tensor_args)
    quat_candidates = quat_by_wxyz / (2.0 * q_abs[..., None].max(flr))
    # choose the best-conditioned quaternion with the largest denominator
    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))


def q_to_axis_angles(q, eps=1e-10):
    q = q.unsqueeze(0) if len(q.shape) == 1 else q
    norm_q = torch.norm(q[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norm_q, q[..., :1])
    angles = 2 * half_angles
    beta = angles.abs() < eps
    s_half_angles = torch.empty_like(angles)
    s_half_angles[~beta] = (
        torch.sin(half_angles[~beta]) / angles[~beta]
    )
    s_half_angles[beta] = (
        0.5 - (angles[beta] * angles[beta]) / 48
    )
    return q[..., 1:] / s_half_angles


def axis_angles_to_q(axis_angles, eps=1e-10):
    axis_angles = axis_angles.unsqueeze(0) if len(axis_angles.shape) == 1 else axis_angles
    angles = torch.norm(axis_angles, p=2, dim=-1, keepdim=True)
    half_angles = angles / 2
    beta = angles.abs() < eps
    s_half_angles = torch.empty_like(angles)
    s_half_angles[~beta] = (
        torch.sin(half_angles[~beta]) / angles[~beta]
    )
    s_half_angles[beta] = (
        0.5 - (angles[beta] * angles[beta]) / 48
    )
    q = torch.cat(
        [torch.cos(half_angles), axis_angles * s_half_angles], dim=-1
    )
    return q


def q_to_euler(q):
    q = q.unsqueeze(0) if len(q.shape) == 1 else q
    w, x, y, z = torch.unbind(q, -1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    t2 = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    euler = torch.stack([roll, pitch, yaw], dim=-1)
    return euler


def euler_to_q(euler):
    euler = euler.unsqueeze(0) if len(euler.shape) == 1 else euler
    roll, pitch, yaw = torch.unbind(euler, -1)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = torch.stack([w, x, y, z], dim=-1)
    return q


def q_convert_xyzw(q):
    w, x, y, z = torch.unbind(q, dim=-1)
    return torch.stack([x, y, z, w], dim=-1)


def q_convert_wxyz(q):
    x, y, z, w = torch.unbind(q, dim=-1)
    return torch.stack([w, x, y, z], dim=-1)