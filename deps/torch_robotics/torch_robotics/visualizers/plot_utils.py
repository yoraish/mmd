from copy import copy

import torch

from torch_robotics.torch_utils.torch_utils import to_numpy


def plot_coordinate_frame(ax, frame, arrow_length=0.1, arrow_alpha=1.0, arrow_linewidth=1.0, tensor_args=None):
    target_pos_np = to_numpy(frame.translation).squeeze()
    frame_target_no_trans = copy(frame)
    frame_target_no_trans.set_translation(torch.zeros_like(frame.translation))

    x_basis = torch.tensor([1, 0, 0], **tensor_args)
    y_basis = torch.tensor([0, 1, 0], **tensor_args)
    z_basis = torch.tensor([0, 0, 1], **tensor_args)

    x_axis_target = to_numpy(frame_target_no_trans.transform_point(x_basis).squeeze())
    y_axis_target = to_numpy(frame_target_no_trans.transform_point(y_basis).squeeze())
    z_axis_target = to_numpy(frame_target_no_trans.transform_point(z_basis).squeeze())

    # x-axis
    ax.quiver(target_pos_np[0], target_pos_np[1], target_pos_np[2],
              x_axis_target[0], x_axis_target[1], x_axis_target[2],
              length=arrow_length, normalize=True, color='red', alpha=arrow_alpha, linewidth=arrow_linewidth)
    # y-axis
    ax.quiver(target_pos_np[0], target_pos_np[1], target_pos_np[2],
              y_axis_target[0], y_axis_target[1], y_axis_target[2],
              length=arrow_length, normalize=True, color='green', alpha=arrow_alpha, linewidth=arrow_linewidth)
    # z-axis
    ax.quiver(target_pos_np[0], target_pos_np[1], target_pos_np[2],
              z_axis_target[0], z_axis_target[1], z_axis_target[2],
              length=arrow_length, normalize=True, color='blue', alpha=arrow_alpha, linewidth=arrow_linewidth)
