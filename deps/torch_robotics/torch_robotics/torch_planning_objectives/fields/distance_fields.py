from abc import ABC, abstractmethod

import einops
import torch
from matplotlib import pyplot as plt

from torch_robotics.torch_kinematics_tree.geometrics.utils import SE3_distance
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
import torch.nn.functional as Functional


class DistanceField(ABC):
    def __init__(self, tensor_args=None):
        self.tensor_args = tensor_args

    def distances(self):
        pass

    def compute_collision(self):
        pass

    @abstractmethod
    def compute_distance(self, *args, **kwargs):
        pass

    def compute_cost(self, q, link_pos, *args, **kwargs):
        q_orig_shape = q.shape
        link_orig_shape = link_pos.shape
        if len(link_orig_shape) == 2:
            h = 1
            b, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b d -> b 1 d")  # add dimension of task space link
        elif len(link_orig_shape) == 3:
            h = 1
            b, t, d = link_orig_shape
        elif len(link_orig_shape) == 4:  # batch, horizon, num_links, 3  # position tensor
            b, h, t, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d -> (b h) t d")
        elif len(link_orig_shape) == 5:  # batch, horizon, num_links, 4, 4  # homogeneous transform tensor
            b, h, t, d, d = link_orig_shape
            link_pos = einops.rearrange(link_pos, "b h t d d -> (b h) t d d")
        else:
            raise NotImplementedError

        # link_tensor_pos
        # position: (batch horizon) x num_links x 3
        cost = self.compute_costs_impl(q, link_pos, *args, **kwargs)

        if cost.ndim == 1:
            cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        # if len(link_orig_shape) == 4 or len(link_orig_shape) == 5:
        #     cost = einops.rearrange(cost, "(b h) -> b h", b=b, h=h)

        return cost

    @abstractmethod
    def compute_costs_impl(self, *args, **kwargs):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


def interpolate_points_v1(points, num_interpolated_points):
    # https://github.com/SamsungLabs/RAMP/blob/c3bd23b2c296c94cdd80d6575390fd96c4f83d83/mppi_planning/cost/collision_cost.py#L89
    """
    Interpolates points to have num_interpolated_points between each pair of points.
    """
    points = Functional.interpolate(points.transpose(-2, -1), size=num_interpolated_points, mode='linear', align_corners=True).transpose(-2, -1)
    return points


# Old implementation
def interpolate_points_v2(points, num_interpolate, link_interpolate_range):
    if num_interpolate > 0:
        link_dim = points.shape[:-1]
        alpha = torch.linspace(0, 1, num_interpolate + 2).type_as(points)[1:num_interpolate + 1]
        alpha = alpha.view(tuple([1] * len(link_dim) + [-1, 1]))  # 1 x 1 x 1 x ... x num_interpolate x 1
        X = points[..., link_interpolate_range[0]:link_interpolate_range[1] + 1, :].unsqueeze(-2)  # batch_dim x num_interp_link x 1 x 3
        X_diff = torch.diff(X, dim=-3)  # batch_dim x (num_interp_link - 1) x 1 x 3
        X_interp = X[..., :-1, :, :] + X_diff * alpha  # batch_dim x (num_interp_link - 1) x num_interpolate x 3
        points = torch.cat([points, X_interp.flatten(-3, -2)], dim=-2)  # batch_dim x (num_link + (num_interp_link - 1) * num_interpolate) x 3
    return points


class EmbodimentDistanceFieldBase(DistanceField):

    def __init__(self,
                 robot,
                 link_idxs_for_collision_checking=None,
                 num_interpolated_points=30,
                 collision_margins=0.,
                 cutoff_margin=0.001,
                 field_type='sdf', clamp_sdf=True,
                 interpolate_link_pos=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert robot is not None, "You need to pass a robot instance to the embodiment distance fields"
        self.robot = robot
        self.link_idxs_for_collision_checking = link_idxs_for_collision_checking
        self.num_interpolated_points = num_interpolated_points
        self.collision_margins = collision_margins
        self.cutoff_margin = cutoff_margin
        self.field_type = field_type
        self.clamp_sdf = clamp_sdf
        self.interpolate_link_pos = interpolate_link_pos

    def compute_embodiment_cost(self, q, link_pos, field_type=None, **kwargs):  # position tensor
        if field_type is None:
            field_type = self.field_type
        if field_type == 'rbf':
            return self.compute_embodiment_rbf_distances(link_pos, **kwargs).sum((-1, -2))
        elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
            margin = self.collision_margins + self.cutoff_margin  # The cutoff margin is the self_collision_margin_vector in the self collision distance field.
            # returns all distances from each link to the environment
            link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]
            margin_minus_sdf = -(self.compute_embodiment_signed_distances(q, link_pos, **kwargs) - margin)
            if self.clamp_sdf:
                if self.clamp_sdf is False:
                    raise ValueError("Debug: clamp intended to be True.")
                clamped_sdf = torch.relu(margin_minus_sdf)
            else:
                clamped_sdf = margin_minus_sdf
            if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
                clamped_sdf = clamped_sdf.max(-2)[0]
            # sum over link points for gradient computation
            return clamped_sdf.sum(-1)
        elif field_type == 'occupancy':
            return self.compute_embodiment_collision(q, link_pos, **kwargs)
            # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
            # return (distances < margin).sum(-1)
        else:
            raise NotImplementedError('field_type {} not implemented'.format(field_type))

    # def compute_embodiment_cost(self, q, link_pos, field_type=None, **kwargs):  # position tensor
    #     if field_type is None:
    #         field_type = self.field_type
    #     if field_type == 'rbf':
    #         return self.compute_embodiment_rbf_distances(link_pos, **kwargs).sum((-1, -2))
    #     elif field_type == 'sdf':  # this computes the negative cost from the DISTANCE FUNCTION
    #         margin = self.collision_margins + self.cutoff_margin
    #         # computes sdf of all links to all objects (or sdf map)
    #         margin_minus_sdf = -(self.compute_embodiment_signed_distances(q, link_pos, **kwargs) - margin)
    #         if self.clamp_sdf:
    #             clamped_sdf = torch.relu(margin_minus_sdf)
    #         else:
    #             clamped_sdf = margin_minus_sdf
    #
    #         if len(clamped_sdf.shape) == 3:  # cover the multiple objects case
    #             clamped_sdf = clamped_sdf.max(-2)[0]
    #
    #         # sum over link points for gradient computation
    #         return clamped_sdf.sum(-1)
    #
    #     elif field_type == 'occupancy':
    #         return self.compute_embodiment_collision(q, link_pos, **kwargs)
    #         # distances = self.self_distances(link_pos, **kwargs)  # batch_dim x (links * (links - 1) / 2)
    #         # return (distances < margin).sum(-1)
    #     else:
    #         raise NotImplementedError('field_type {} not implemented'.format(field_type))

    def compute_costs_impl(self, q, link_pos, **kwargs):
        # position link_pos tensor # batch x num_links x 3
        # interpolate to approximate link spheres
        if self.robot.grasped_object is not None:
            n_grasped_object_points = self.robot.grasped_object.n_base_points_for_collision
            link_pos_robot = link_pos[..., :-n_grasped_object_points, :]
            link_pos_grasped_object = link_pos[..., -n_grasped_object_points:, :]
        else:
            link_pos_robot = link_pos

        link_pos_robot = link_pos_robot[..., self.link_idxs_for_collision_checking, :]
        if self.interpolate_link_pos:
            # select the robot links used for collision checking
            link_pos = interpolate_points_v1(link_pos_robot, self.num_interpolated_points)

        # stack collision points from grasped object
        # these points do not need to be interpolated
        if self.robot.grasped_object is not None:
            link_pos = torch.cat((link_pos, link_pos_grasped_object), dim=-2)

        embodiment_cost = self.compute_embodiment_cost(q, link_pos, **kwargs)
        return embodiment_cost

    def compute_distance(self, q, link_pos, **kwargs):
        raise NotImplementedError
        link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points)
        self_distances = self.compute_embodiment_signed_distances(q, link_pos, **kwargs).min(-1)[0]  # batch_dim
        return self_distances

    def zero_grad(self):
        pass
        # raise NotImplementedError

    @abstractmethod
    def compute_embodiment_rbf_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_signed_distances(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_embodiment_collision(self, *args, **kwargs):
        raise NotImplementedError


class CollisionSelfField(EmbodimentDistanceFieldBase):

    def __init__(self, *args, idxs_links_distance_matrix=None, **kwargs):
        super().__init__(*args, collision_margins=0., **kwargs)
        self.idxs_links_distance_matrix = idxs_links_distance_matrix
        self.idxs_links_distance_matrix_tuple = tuple(zip(*idxs_links_distance_matrix))

    def compute_embodiment_rbf_distances(self, link_pos, **kwargs):  # position tensor
        raise NotImplementedError
        margin = kwargs.get('margin', self.cutoff_margin)
        rbf_distance = torch.exp(torch.square(link_pos.unsqueeze(-2) -
                                              link_pos.unsqueeze(-3)).sum(-1) / (-margin ** 2 * 2))
        return rbf_distance

    def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):  # position tensor
        """
        Computes the distances between all pairs of links.
        The distances are computed as the L2 norm between the link positions, with the position of each link being
        a point in 3D space.
        """
        if link_pos.shape[-2] == 1:
            # if there is only one link, the self distance is very large
            # implementation guarantees gradient computation
            return torch.abs(link_pos).sum(-1) * 1e9
        dist_mat = torch.linalg.norm(link_pos.unsqueeze(-2) - link_pos.unsqueeze(-3), dim=-1)  # batch_dim x links x links

        # select lower triangular -- distance between link points
        # lower_indices = torch.tril_indices(dist_mat.shape[-1], dist_mat.shape[-1], offset=-1).unbind()
        # distances = dist_mat[..., lower_indices[0], lower_indices[1]]  # batch_dim x (links * (links - 1) / 2)

        # select only distances between pairs of specified links
        distances = dist_mat[..., self.idxs_links_distance_matrix_tuple[0], self.idxs_links_distance_matrix_tuple[1]]

        return distances

    def compute_embodiment_collision(self, q, link_pos, **kwargs):  # position tensor
        margin = kwargs.get('margin', self.cutoff_margin)
        link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]
        distances = self.compute_embodiment_signed_distances(q, link_pos, **kwargs)
        any_self_collision = torch.any(distances < margin, dim=-1)
        return any_self_collision


def reshape_q(q):
    q_orig_shape = q.shape
    if len(q_orig_shape) == 2:
        h = 1
        b, d = q_orig_shape
    elif len(q_orig_shape) == 3:
        b, h, d = q_orig_shape
        q = einops.rearrange(q, "b h d -> (b h) d")
    else:
        raise NotImplementedError
    return q, q_orig_shape, b, h, d


# class CollisionSelfFieldWrapperSTORM(EmbodimentDistanceFieldBase):
#
#     def __init__(self, robot, weights_fname, n_joints, *args, **kwargs):
#         super().__init__(robot, *args, collision_margins=0., interpolate_link_pos=False, **kwargs)
#         from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet
#         self.robot_self_collision_net = RobotSelfCollisionNet(n_joints)
#         self.robot_self_collision_net.load_weights(weights_fname, self.tensor_args)
#
#     def compute_costs_impl(self, q, link_pos, **kwargs):
#         embodiment_cost = self.compute_embodiment_cost(q, link_pos, **kwargs)
#         return embodiment_cost
#
#     def compute_embodiment_rbf_distances(self, *args, **kwargs):  # position tensor
#         raise NotImplementedError
#
#     def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):  # position tensor
#         q, q_orig_shape, b, h, d = reshape_q(q)
#
#         # multiply by -1, because according to the paper (page 6, https://arxiv.org/pdf/2104.13542.pdf)
#         # "Distance is positive when two links are penetrating and negative when not colliding"
#         sdf_self_collision = -1. * self.robot_self_collision_net.compute_signed_distance(q, with_grad=True)
#
#         return sdf_self_collision
#
#     def compute_embodiment_collision(self, q, link_pos, **kwargs):  # position tensor
#         q, q_orig_shape, b, h, d = reshape_q(q)
#
#         # multiply by -1, because according to the paper (page 6, https://arxiv.org/pdf/2104.13542.pdf)
#         # "Distance is positive when two links are penetrating and negative when not colliding"
#         sdf_self_collision = -1. * self.robot_self_collision_net.compute_signed_distance(q, with_grad=True)
#
#         if len(q_orig_shape) == 3:
#             sdf_self_collision = einops.rearrange(sdf_self_collision, "(b h) 1 -> b h", b=b, h=h)
#
#         any_self_collision = sdf_self_collision < -0.05  # trained on 0.02
#         return any_self_collision


class CollisionObjectBase(EmbodimentDistanceFieldBase):

    def __init__(self, *args, link_margins_for_object_collision_checking_tensor=None, **kwargs):
        super().__init__(*args, collision_margins=link_margins_for_object_collision_checking_tensor, **kwargs)

    def compute_embodiment_rbf_distances(self, link_pos, **kwargs):  # position tensor
        raise NotImplementedError
        margin = kwargs.get('margin', self.margin)
        rbf_distance = torch.exp(torch.square(self.object_signed_distances(link_pos, **kwargs)) / (-margin ** 2 * 2))
        return rbf_distance

    def compute_embodiment_signed_distances(self, q, link_pos, **kwargs):
        return self.object_signed_distances(link_pos, **kwargs)

    def compute_embodiment_collision(self, q, link_pos, **kwargs):
        # position tensor
        margin = kwargs.get('margin', self.collision_margins + self.cutoff_margin)
        link_pos = link_pos[..., self.link_idxs_for_collision_checking, :]
        signed_distances = self.object_signed_distances(link_pos, **kwargs)
        collisions = signed_distances < margin
        # reduce over points (dim -1) and over objects (dim -2)
        any_collision = torch.any(torch.any(collisions, dim=-1), dim=-1)
        return any_collision

    @abstractmethod
    def object_signed_distances(self, *args, **kwargs):
        raise NotImplementedError


class CollisionObjectDistanceField(CollisionObjectBase):

    def __init__(self,
                 *args,
                 df_obj_list_fn=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.df_obj_list_fn = df_obj_list_fn

    def object_signed_distances(self, link_pos, **kwargs):
        if self.df_obj_list_fn is None:
            return torch.inf
        df_obj_list = self.df_obj_list_fn()
        link_dim = link_pos.shape[:-1]
        link_pos = link_pos.reshape(-1, link_pos.shape[-1])  # flatten batch_dim and links
        dfs = []
        for df in df_obj_list:
            dfs.append(df.compute_signed_distance(link_pos).view(link_dim))  # df() returns batch_dim x links
        return torch.stack(dfs, dim=-2)  # batch_dim x num_sdfs x links


class CollisionWorkspaceBoundariesDistanceField(CollisionObjectBase):

    def __init__(self, *args, ws_bounds_min=None, ws_bounds_max=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws_min = ws_bounds_min
        self.ws_max = ws_bounds_max

    def object_signed_distances(self, link_pos, **kwargs):
        signed_distances_bounds_min = link_pos - self.ws_min
        signed_distances_bounds_min = torch.sign(signed_distances_bounds_min) * torch.abs(signed_distances_bounds_min)
        signed_distances_bounds_max = self.ws_max - link_pos
        signed_distances_bounds_max = torch.sign(signed_distances_bounds_max) * torch.abs(signed_distances_bounds_max)
        signed_distances_bounds = torch.cat((signed_distances_bounds_min, signed_distances_bounds_max), dim=-1)
        return signed_distances_bounds.transpose(-2, -1)    # batch_dim x num_sdfs x links


class EESE3DistanceField(DistanceField):

    def __init__(self, target_H, w_pos=1., w_rot=1., square=True, **kwargs):
        super().__init__(**kwargs)
        self.target_H = target_H
        self.square = square
        self.w_pos = w_pos
        self.w_rot = w_rot

    def update_target(self, target_H):
        self.target_H = target_H

    def compute_distance(self, link_tensor):
        # homogeneous transformation link_tensor # batch x num_links x 4 x 4
        # -1: get EE as last link  # TODO - get EE from its name id
        return SE3_distance(link_tensor[..., -1, :, :], self.target_H, w_pos=self.w_pos, w_rot=self.w_rot)

    def compute_costs_impl(self, q, link_tensor, **kwargs):  # position tensor
        dist = self.compute_distance(link_tensor).squeeze()
        if self.square:
            dist = torch.square(dist)
        return dist

    def zero_grad(self):
        raise NotImplementedError

if __name__ == '__main__':
    raise NotImplementedError
    import time
    mesh_file = 'models/chair.obj'
    mesh = MeshDistanceField(mesh_file)
    bounds = np.array(mesh.mesh.bounds)
    print(np.linalg.norm(bounds[1] - bounds[0]))
    print(mesh.mesh.centroid)
    points = torch.rand(100, 3)
    link_tensor = torch.rand(100, 10, 4, 4)
    start = time.time()
    distances = mesh.compute_distance(points)
    costs = mesh.compute_cost(link_tensor)
    print(time.time() - start)
