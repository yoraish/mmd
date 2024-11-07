import abc

import torch
from urdf_parser_py.urdf import Box

from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import to_numpy


class GraspedObject(ObjectField):

    def __init__(self, primitive_fields, **kwargs):
        # pos, ori - position and orientation are specified wrt to the end-effector link

        # Only one primitive type
        assert len(primitive_fields) == 1

        super().__init__(primitive_fields, **kwargs)

        # Geometry URDF
        self.geometry_urdf = self.get_geometry_urdf()

    def get_geometry_urdf(self):
        primitive_field = self.fields[0]

        if isinstance(primitive_field, MultiBoxField):
            size = primitive_field.sizes[0]
            return Box(to_numpy(size))
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def get_base_points_for_collision(self):
        raise NotImplementedError


class GraspedObjectPandaBox(GraspedObject):

    def __init__(self, tensor_args=None, **kwargs):
        # One box
        primitive_fields = [
            MultiBoxField(torch.zeros(3, **tensor_args).view(1, -1),
                          torch.tensor([0.05, 0.05, 0.15], **tensor_args).view(1, -1),
                          tensor_args=tensor_args)
        ]

        # position and orientation wrt to the robots's end-effector link -> for panda reference_frame='panda_hand'
        pos = torch.tensor([0., 0., 0.11], **tensor_args)
        ori = torch.tensor([0, 0.7071081, 0, 0.7071055], **tensor_args)

        super().__init__(
            primitive_fields,
            name='GraspedObjectPandaBox',
            pos=pos, ori=ori, reference_frame='panda_hand',
            **kwargs)

        self.base_points_for_collision = self.get_base_points_for_collision()
        self.n_base_points_for_collision = len(self.base_points_for_collision)

    def get_base_points_for_collision(self):
        # points on vertices and centers of faces
        size = self.fields[0].sizes[0]
        x, y, z = size
        vertices = torch.tensor(
            [
                [x/2, y/2, -z/2],
                [x/2, -y/2, -z/2],
                [-x/2, -y/2, -z/2],
                [-x/2, y/2, -z/2],
                [x/2, y/2, z/2],
                [x/2, -y/2, z/2],
                [-x/2, -y/2, z/2],
                [-x/2, y/2, z/2],
            ],
            **self.tensor_args)

        faces = torch.tensor(
            [
                [x/2, 0, 0],
                [0, -y/2, 0],
                [-x/2, 0, 0],
                [0, y/2, 0],
                [0, 0, z/2],
                [0, 0, -z/2],
            ],
            **self.tensor_args)

        points = torch.cat((vertices, faces), dim=0)
        return points
