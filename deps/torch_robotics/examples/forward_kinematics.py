import torch
import time

from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda, DifferentiableUR10, \
    DifferentiableTiagoDualHoloMove, DifferentiableShadowHand, DifferentiableAllegroHand, DifferentiableHabitatStretch
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA


if __name__ == "__main__":
    seed = 1
    fix_random_seed(seed)

    batch_size = 10
    device = "cpu"
    # device = "cuda:0"

    print("\n===========================Panda Model===============================")
    diff_panda = DifferentiableFrankaPanda(device=device)
    diff_panda.print_link_names()
    print(diff_panda.get_joint_limits())
    print(diff_panda._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_panda._n_dofs).to(device).requires_grad_(True)
        data = diff_panda.compute_forward_kinematics_all_links(q)
    print(f"Computational Time {t.elapsed:.4f}")

    print("\n===========================UR10 Model===============================")
    diff_ur10 = DifferentiableUR10(device=device)
    diff_ur10.print_link_names()
    print(diff_ur10.get_joint_limits())
    print(diff_ur10._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_ur10._n_dofs).to(device).requires_grad_(True)
        data = diff_ur10.compute_forward_kinematics_all_links(q)
        print(data.shape)
    print(f"Computational Time {t.elapsed:.4f}")

    print("\n===========================Habitat Stretch Model===============================")
    diff_str = DifferentiableHabitatStretch(device=device)
    diff_str.print_link_names()
    print(diff_str.get_joint_limits())
    print(diff_str._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_str._n_dofs).to(device).requires_grad_(True)
        data = diff_str.compute_forward_kinematics_all_links(q)
        print(data.shape)
    print(f"Computational Time {t.elapsed:.4f}")

    print("\n===========================Tiago Model===============================")
    diff_tiago = DifferentiableTiagoDualHoloMove(device=device)
    diff_tiago.print_link_names()
    print(diff_tiago.get_joint_limits())
    print(diff_tiago._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_tiago._n_dofs).to(device).requires_grad_(True)
        data = diff_tiago.compute_forward_kinematics_all_links(q)
        print(data.shape)
    print(f"Computational Time {t.elapsed:.4f}")

    print("\n===========================Shadow Hand Model===============================")
    hand = DifferentiableShadowHand(device=device)
    hand.print_link_names()
    print(hand.get_joint_limits())
    print(hand._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, hand._n_dofs).to(device).requires_grad_(True)
        data = hand.compute_forward_kinematics_all_links(q)
        print(data.shape)
    print(f"Computational Time {t.elapsed:.4f}")

    print("\n===========================Allegro Hand Model===============================")
    hand = DifferentiableAllegroHand(device=device)
    hand.print_link_names()
    print(hand.get_joint_limits())
    print(hand._n_dofs)
    with TimerCUDA() as t:
        q = torch.rand(batch_size, hand._n_dofs).to(device).requires_grad_(True)
        data = hand.compute_forward_kinematics_all_links(q)
        print(data.shape)
    print(f"Computational Time {t.elapsed:.4f}")
