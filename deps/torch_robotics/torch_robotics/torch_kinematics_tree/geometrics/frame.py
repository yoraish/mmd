from __future__ import annotations
from typing import Optional
import torch
import math

from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_to_rotation_matrix
from torch_robotics.torch_kinematics_tree.geometrics.utils import vector3_to_skew_symm_matrix, multiply_inv_transform, \
    multiply_transform



class Frame(object):

    def __init__(self,
                 rot: Optional[torch.Tensor] = None,
                 trans: Optional[torch.Tensor] = None,
                 pose: Optional[torch.Tensor] = None,
                 device='cpu'):

        self.device = device

        if rot is None:
            self._rot = torch.eye(3, device=device).unsqueeze(0)
        else:
            self._rot = rot.to(device)
            if len(self._rot.shape) == 2:
                self._rot = self._rot.unsqueeze(0)

        if trans is None:
            self._trans = torch.zeros(1, 3, device=device)
        else:
            self._trans = trans.to(device)
            if len(self._trans.shape) == 1:
                self._trans = self._trans.unsqueeze(0)

        if (pose is not None):
            self.set_pose(pose)
        assert self._trans.shape[0] == self._rot.shape[0]
        self.batch_size = self._trans.shape[0]

    def set_pose(self, pose: torch.Tensor) -> None:
        """
        Args:
            pose[0, :]: x, y, z, qw, qx, qy, qz
        """
        if pose.ndim == 1:
            pose = pose.unsqueeze(0)
        self._trans = pose[:, :3].clone().to(self.device)
        self._rot = q_to_rotation_matrix(pose[:, 3:]).to(self.device)

    def set_translation(self, t: torch.Tensor) -> None:
        self._trans = t.to(self.device)

    def set_rotation(self, rot: torch.Tensor) -> None:
        self._rot = rot.to(self.device)

    def inverse(self) -> Frame:
        rot_transpose = self._rot.transpose(-2, -1)
        return Frame(
            rot_transpose,
            -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2),
            device=self.device)

    def multiply_transform(self, frame: Frame) -> Frame:
        new_rot, new_trans = multiply_transform(
            self._rot, self._trans, frame.rotation,
            frame.translation)
        return Frame(new_rot, new_trans, device=self.device)

    def multiply_inv_transform(self, frame: Frame) -> Frame:
        new_rot, new_trans = multiply_inv_transform(
            frame.rotation,
            frame.translation, self._rot, self._trans)
        return Frame(new_rot,
                                   new_trans,
                                   device=self.device)

    def trans_cross_rot(self) -> torch.Tensor:
        return vector3_to_skew_symm_matrix(self._trans) @ self._rot

    def get_transform_matrix(self) -> torch.Tensor:
        mat = torch.eye(4, device=self.device).repeat(self.batch_size, 1, 1)
        mat[:, :3, :3] = self._rot
        mat[:, :3, 3] = self._trans
        return mat

    def get_quaternion(self) -> torch.Tensor:
        M = torch.zeros((self.batch_size, 4, 4)).to(self._rot.device)
        M[:, :3, :3] = self._rot
        M[:, :3, 3] = self._trans
        M[:, 3, 3] = 1
        q = torch.empty((self.batch_size, 4)).to(self._rot.device)
        t = torch.einsum('bii->b', M)
        for n in range(self.batch_size):
            tn = t[n]
            if tn > M[n, 3, 3]:
                q[n, 3] = tn
                q[n, 2] = M[n, 1, 0] - M[n, 0, 1]
                q[n, 1] = M[n, 0, 2] - M[n, 2, 0]
                q[n, 0] = M[n, 2, 1] - M[n, 1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[n, 1, 1] > M[n, 0, 0]:
                    i, j, k = 1, 2, 0
                if M[n, 2, 2] > M[n, i, i]:
                    i, j, k = 2, 0, 1
                tn = M[n, i, i] - (M[n, j, j] + M[n, k, k]) + M[n, 3, 3]
                q[n, i] = tn
                q[n, j] = M[n, i, j] + M[n, j, i]
                q[n, k] = M[n, k, i] + M[n, i, k]
                q[n, 3] = M[n, k, j] - M[n, j, k]

            q[n, :] *= 0.5 / math.sqrt(tn * M[n, 3, 3])
        return q

    def transform_point(self, point: torch.Tensor) -> torch.Tensor:
        new_point = ((self._rot @ point.unsqueeze(0).transpose(-1, -2)).transpose(-1, -2)) + self._trans.unsqueeze(-2)
        return new_point

    def get_euler(self) -> torch.Tensor:
        return torch.atan2(self._rot[:, 2, 1], self._rot[:, 2, 2]), torch.asin(-self._rot[:, 2, 0]), torch.atan2(self._rot[:, 1, 0], self._rot[:, 0, 0])

    @property
    def rotation(self) -> torch.Tensor:
        return self._rot

    @property
    def translation(self) -> torch.Tensor:
        return self._trans


if __name__ == '__main__':
    # test Frame
    frame = Frame()
    print(frame.get_transform_matrix())
    print(frame.get_quaternion())
    print(frame.get_euler())
