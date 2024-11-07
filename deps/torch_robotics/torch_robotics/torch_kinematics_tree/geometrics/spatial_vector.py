from __future__ import annotations
from typing import Optional
import torch
from torch_robotics.torch_kinematics_tree.geometrics.utils import cross_product, to_torch_2d_min
from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame


def x_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = to_torch_2d_min(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.ones(batch_size)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 1, 2] = -torch.sin(angle)
    R[:, 2, 1] = torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def y_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = to_torch_2d_min(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 2] = torch.sin(angle)
    R[:, 1, 1] = torch.ones(batch_size)
    R[:, 2, 0] = -torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def z_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = to_torch_2d_min(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size)
    return R


class MotionVec(object):
    def __init__(
        self,
        lin_motion: Optional[torch.Tensor] = None,
        ang_motion: Optional[torch.Tensor] = None,
        device='cpu',
    ):
        if lin_motion is None or ang_motion is None:
            device = device
        self.lin = (
            lin_motion if lin_motion is not None else torch.zeros((1, 3), device=device)
        )
        self.ang = (
            ang_motion if ang_motion is not None else torch.zeros((1, 3), device=device)
        )

    def add_motion_vec(self, mv: MotionVec) -> MotionVec:
        """
        Args:
            mv: spatial motion vector
        Returns:
            the sum of motion vectors
        """

        return MotionVec(self.lin + mv.lin, self.ang + mv.ang)

    def cross_motion_vec(self, mv: MotionVec) -> MotionVec:
        """
        Args:
            mv: spatial motion vector
        Returns:
            the cross product between motion vectors
        """
        new_ang = cross_product(self.ang, mv.ang)
        new_lin = cross_product(self.ang, mv.lin) + cross_product(self.lin, mv.ang)
        return MotionVec(new_lin, new_ang)

    def transform(self, transform: Frame) -> MotionVec:
        """
        Args:
            transform: a coordinate transform object
        Returns:
            the motion vector (self) transformed by the coordinate transform
        """
        new_ang = (transform.rotation @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin = (transform.trans_cross_rot() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin += (transform.rotation @ self.lin.unsqueeze(2)).squeeze(2)
        return MotionVec(new_lin, new_ang)

    def get_vector(self):
        return torch.cat([self.ang, self.lin], dim=1)

    def dot(self, mv):
        tmp1 = torch.sum(self.ang * mv.ang, dim=-1)
        tmp2 = torch.sum(self.lin * mv.lin, dim=-1)
        return tmp1 + tmp2

