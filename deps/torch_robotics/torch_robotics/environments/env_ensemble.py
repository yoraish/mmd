from typing import Dict
from matplotlib import pyplot as plt

import torch

from torch_robotics.environments import *
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvEnsemble(EnvBase):
    def __init__(self,
                 envs: Dict[int, EnvBase],
                 transforms: Dict[int, torch.Tensor],
                 tensor_args=None,
                 **kwargs):
        self.envs = envs
        self.transforms = transforms  # @Note: transforms are the relative positions of the envs, where the first env
        # is the absolut reference

        self.env_centers = {k: self.transforms[k] for k in self.envs.keys()}
        self.env_limits = {k: self.envs[k].limits for k in self.envs.keys()}

        # loop on envs and transforms, and get the big map limits
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS
        absolute_limits = torch.zeros(self.env_limits[0].shape, **tensor_args)
        for k in self.envs.keys():
            limits = self.env_limits[k]
            center = self.env_centers[k]
            limits = limits + center
            absolute_limits[0] = torch.min(absolute_limits[0], limits[0])
            absolute_limits[1] = torch.max(absolute_limits[1], limits[1])

        obj_list = []
        for k, env in self.envs.items():
            obj_k_list = env.get_obj_list()
            for obj in obj_k_list:
                # translate the object to the absolute position obj.reference_frame = self.env_centers[k] #TODO: It
                #  would be nice to have a reference frame for the object and thats enough, but currently there is no
                #  actual use for it in EnvBase
                if obj.pos.shape[0] == 3 and self.env_centers[k].shape[0] == 2:
                    obj.pos[:2] += self.env_centers[k]
                else:
                    obj.pos += self.env_centers[k]
                obj_list.append(obj)

        super().__init__(limits=absolute_limits,
                         obj_fixed_list=obj_list,
                         tensor_args=tensor_args,
                         **kwargs)


if __name__ == '__main__':
    env0 = EnvEmpty2D(tensor_args=DEFAULT_TENSOR_ARGS)

    env1 = EnvConveyor2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )

    env2 = EnvSquare2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )

    env3 = EnvDense2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )

    envs = {0: env0, 1: env1, 2: env2, 3: env3}

    transforms = {0: torch.tensor([0., 0.], **DEFAULT_TENSOR_ARGS),
                  1: torch.tensor([2., 0.], **DEFAULT_TENSOR_ARGS),
                  2: torch.tensor([2., -2.], **DEFAULT_TENSOR_ARGS),
                  3: torch.tensor([0., -2.], **DEFAULT_TENSOR_ARGS)}

    env = EnvEnsemble(envs, transforms, tensor_args=DEFAULT_TENSOR_ARGS)

    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()