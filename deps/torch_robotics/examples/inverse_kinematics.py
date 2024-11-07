from copy import copy

import matplotlib.pyplot as plt
import torch

from torch_robotics.torch_kinematics_tree.geometrics.frame import Frame
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.spatial_vector import y_rot, z_rot
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_rot_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


if __name__ == "__main__":
    seed = 0
    fix_random_seed(seed)

    batch_size = 10
    device = 'cuda'
    # device = 'cpu'
    tensor_args = dict(device=device, dtype=torch.float32)

    pos_target = torch.tensor([0.2, 0.4, 0.1], **tensor_args)
    rot_target = (z_rot(-torch.tensor(torch.pi/2)) @ y_rot(-torch.tensor(torch.pi))).to(**tensor_args)
    frame_target = Frame(rot=rot_target, trans=pos_target, device=device)
    H_target = frame_target.get_transform_matrix()  # set translation and orientation of target here

    print("===========================Panda Model===============================")
    # Panda Kinematics
    diff_panda = DifferentiableFrankaPanda(gripper=False, device=device)

    with TimerCUDA() as t:
        q_ik, idx_valid = diff_panda.inverse_kinematics(
            H_target, link_name='ee_link', batch_size=batch_size, max_iters=500, lr=2e-1, se3_eps=5e-2,
            eps_joint_lim=torch.pi/64,
            # print_freq=-1,
            debug=False
        )
    print(f"\nIK time: {t.elapsed:.3f} sec")

    print(f'idx_valid: {idx_valid.nelement()}/{batch_size}')

    ############################################################################################################
    # plot the result
    fig, ax = create_fig_and_axes(3)
    ax.set_aspect('equal')
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_zlim(-0.5, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Origin frame
    frame_origin = Frame(device=device)
    ax.plot(0, 0, 0, color='b', marker='D', markersize=10, zorder=100)
    plot_coordinate_frame(ax, frame_origin, arrow_length=0.1, arrow_linewidth=3., tensor_args=tensor_args)

    # Target frame
    target_pos_np = to_numpy(pos_target)
    ax.plot(target_pos_np[0], target_pos_np[1], target_pos_np[2], 'r*', markersize=20, zorder=100)
    plot_coordinate_frame(ax, frame_target, arrow_length=0.15, arrow_linewidth=3., tensor_args=tensor_args)

    # Robot
    for q in q_ik[idx_valid]:
        skeleton = get_skeleton_from_model(diff_panda, q, diff_panda.get_link_names())  # visualize IK solution
        skeleton.draw_skeleton(ax=ax, color='blue')

        # draw EE frame
        H_EE = diff_panda.compute_forward_kinematics_all_links(q.unsqueeze(0), link_list=['ee_link'])
        frame_EE = Frame(
            rot=link_rot_from_link_tensor(H_EE).squeeze(),
            trans=link_pos_from_link_tensor(H_EE).squeeze(),
            device=device
        )
        plot_coordinate_frame(ax, frame_EE, arrow_length=0.1, arrow_alpha=0.5, tensor_args=tensor_args)

    plt.show()
