import torch
from networkx.readwrite import json_graph
from urdf_parser_py.urdf import URDF

from torch_robotics.torch_kinematics_tree.utils.files import get_mjcf_path

torch.set_default_dtype(torch.float32)

JOINT_NAME_MAP = {
    'hinge': 'revolute',
    'slide': 'prismatic'
}


class RobotModel(object):

    def __init__(self, model_path, device="cpu") -> None:
        self.model_path = model_path
        self._device = torch.device(device)

    def find_joint_of_body(self, body_name):
        pass

    def get_name_of_parent_body(self, body_name):
        pass
    
    def get_body_parameters(self, i, link):
        pass


class MJCFRobotModel(RobotModel):
    def __init__(self, model_path, device="cpu"):
        super().__init__(model_path, device=device)
        from dm_control import mjcf
        self.robot = mjcf.from_file(self.model_path)
        self._device = torch.device(device)
        bodies = self.robot.find_all("body")
        self.links = [e for e in bodies]
        self._bodies = {e.name: e for e in bodies}
        self._joints = {e.name: e for e in self.robot.find_all("joint")}
        # TODO: implement default logics of mujoco
        # TODO: tendon forward kinematic?
        # NOTE: this class does not work correctly now
        # self._actuators = {e.name: e for e in self.robot.find_all("actuator")}
        node_list, edge_list = [], []
        for b in self.robot.worldbody.body:
            self.build_topology(b, node_list, edge_list)
        self.topology = json_graph.node_link_graph({'nodes': node_list, 'links': edge_list}, directed=True)

    def build_topology(self, body, node_list, edge_list):  # since there is no parent tag for joint
        node_list.append({'id': body.name})
        if not body.body:
            return
        for b in body.body:
            edge_list.append({
                'source': body.name,
                'target': b.name
            })
            self.build_topology(b, node_list, edge_list)

    def find_joint_of_body(self, body_name):
        if body_name not in self._bodies:
            return -1
        return self._bodies[body_name].joint

    def get_name_of_parent_body(self, body_name):
        if body_name not in self._bodies:
            return
        return [b for b in self.topology.predecessors(body_name)]

    def get_body_parameters(self, i, link):
        body_params = {}
        body_params["joint_id"] = i
        body_params["link_name"] = link.name

        if i == 0:
            rot_angles = torch.zeros(3, device=self._device)
            trans = torch.zeros(3, device=self._device)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), device=self._device)
        else:
            link_name = link.name
            joint = self.find_joint_of_body(link_name)
            if not joint:
                rot_angles = torch.zeros(3, device=self._device)
                trans = torch.zeros(3, device=self._device)
                joint_name = link_name
                joint_type = "fixed"
                joint_limits = None
                joint_damping = None
                joint_axis = torch.zeros((1, 3), device=self._device)
            else:
                joint = joint[0]
                joint_name = joint.name

                rot_angles = torch.zeros(3, device=self._device)  # joint angle is not specified in mujoco
                if joint.pos is not None:
                    trans = torch.tensor(
                        joint.pos, dtype=torch.float32, device=self._device
                    )
                else:
                    trans = torch.zeros(3, device=self._device)
                if joint.type is None:
                    joint_type = 'revolute'
                else:
                    joint_type = JOINT_NAME_MAP[joint.type]
                if joint.range is None:
                    joint_limits = None
                else:
                    joint_limits = {
                        "lower": joint.range[0],
                        "upper": joint.range[1],
                    }
                if joint.damping is None:
                    joint_damping = None
                else:
                    joint_damping = torch.tensor([joint.damping], device=self._device)
                joint_axis = torch.tensor(
                    joint.axis, dtype=torch.float32, device=self._device
                ).reshape(1, 3)

        body_params["rot_angles"] = rot_angles
        body_params["trans"] = trans
        body_params["joint_name"] = joint_name
        body_params["joint_type"] = joint_type
        body_params["joint_limits"] = joint_limits
        body_params["joint_damping"] = joint_damping
        body_params["joint_axis"] = joint_axis

        if link.inertial is not None:
            mass = torch.tensor(
                [link.inertial.mass], dtype=torch.float32, device=self._device
            )
            if link.pos is not None:
                pos = link.pos
            else:
                pos = [0, 0, 0]
            com = (
                    torch.tensor(
                        pos,
                        dtype=torch.float32,
                        device=self._device,
                    )
                    .reshape((1, 3))
                    .to(self._device)
                )
            if link.inertial.diaginertia is not None:
                inert_mat = torch.zeros((3, 3), device=self._device)
                inert_mat[0, 0] = link.inertial.diaginertia[0]
                inert_mat[1, 1] = link.inertial.diaginertia[1]
                inert_mat[2, 2] = link.inertial.diaginertia[2]
            else:
                print(
                    "Warning: No inertia information for link: {}, setting inertia matrix to identity.".format(
                    link.name
                ))
                inert_mat = torch.eye(3, device=self._device)

            inert_mat = inert_mat.unsqueeze(0)
            body_params["mass"] = mass
            body_params["com"] = com
            body_params["inertia_mat"] = inert_mat
        else:
            body_params["mass"] = torch.ones((1,), device=self._device)
            body_params["com"] = torch.zeros((1, 3), device=self._device)
            body_params["inertia_mat"] = torch.eye(3, 3, device=self._device).unsqueeze(
                0
            )
            print(
                "Warning: No dynamics information for link: {}, setting all inertial properties to 1.".format(
                    link.name
                )
            )

        return body_params


class URDFRobotModel(RobotModel):
    def __init__(self, model_path, device="cpu"):
        super().__init__(model_path, device=device)
        self.robot = URDF.from_xml_file(self.model_path)
        self.links = self.robot.links
        self._device = torch.device(device)

    def find_joint_of_body(self, body_name):
        for (i, joint) in enumerate(self.robot.joints):
            if joint.child == body_name:
                return i
        return -1

    def get_name_of_parent_body(self, link_name):
        jid = self.find_joint_of_body(link_name)
        joint = self.robot.joints[jid]
        return joint.parent

    def get_body_parameters(self, i, link):
        body_params = {}
        body_params["joint_id"] = i
        body_params["link_name"] = link.name

        if i == 0:
            rot_angles = torch.zeros(3, device=self._device)
            trans = torch.zeros(3, device=self._device)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), device=self._device)
        else:
            link_name = link.name
            jid = self.find_joint_of_body(link_name)
            joint = self.robot.joints[jid]
            joint_name = joint.name
            # find joint that is the "child" of this body according to urdf

            rot_angles = torch.tensor(
                joint.origin.rotation, dtype=torch.float32, device=self._device
            )
            trans = torch.tensor(
                joint.origin.position, dtype=torch.float32, device=self._device
            )
            joint_type = joint.type
            joint_limits = None
            joint_damping = torch.zeros(1, device=self._device)
            if joint.axis is not None:
                joint_axis = torch.tensor(
                    joint.axis, dtype=torch.float32, device=self._device
                ).reshape(1, 3)
            else:
                joint_axis = torch.zeros((1, 3), device=self._device)
            if joint_type != "fixed" and joint.limit is not None:
                joint_limits = {
                    "effort": joint.limit.effort,
                    "lower": joint.limit.lower,
                    "upper": joint.limit.upper,
                    "velocity": joint.limit.velocity,
                }
                try:
                    joint_damping = torch.tensor(
                        [joint.dynamics.damping],
                        dtype=torch.float32,
                        device=self._device,
                    )
                except:
                    joint_damping = torch.zeros(1, device=self._device)

        body_params["rot_angles"] = rot_angles
        body_params["trans"] = trans
        body_params["joint_name"] = joint_name
        body_params["joint_type"] = joint_type
        body_params["joint_limits"] = joint_limits
        body_params["joint_damping"] = joint_damping
        body_params["joint_axis"] = joint_axis

        if link.inertial is not None:
            mass = torch.tensor(
                [link.inertial.mass], dtype=torch.float32, device=self._device
            )
            if link.inertial.origin is not None:
                pos = link.inertial.origin.position
            else:
                pos = [0, 0, 0]
            com = (
                    torch.tensor(
                        pos,
                        dtype=torch.float32,
                        device=self._device,
                    )
                    .reshape((1, 3))
                    .to(self._device)
                )
            if link.inertial.inertia is not None:
                inert_mat = torch.zeros((3, 3), device=self._device)
                inert_mat[0, 0] = link.inertial.inertia.ixx
                inert_mat[0, 1] = link.inertial.inertia.ixy
                inert_mat[0, 2] = link.inertial.inertia.ixz
                inert_mat[1, 0] = link.inertial.inertia.ixy
                inert_mat[1, 1] = link.inertial.inertia.iyy
                inert_mat[1, 2] = link.inertial.inertia.iyz
                inert_mat[2, 0] = link.inertial.inertia.ixz
                inert_mat[2, 1] = link.inertial.inertia.iyz
                inert_mat[2, 2] = link.inertial.inertia.izz
            else:
                print(
                    "Warning: No inertia information for link: {}, setting inertia matrix to identity.".format(
                    link.name
                ))
                inert_mat = torch.eye(3, device=self._device)

            inert_mat = inert_mat.unsqueeze(0)
            body_params["mass"] = mass
            body_params["com"] = com
            body_params["inertia_mat"] = inert_mat
        else:
            body_params["mass"] = torch.ones((1,), device=self._device)
            body_params["com"] = torch.zeros((1, 3), device=self._device)
            body_params["inertia_mat"] = torch.eye(3, 3, device=self._device).unsqueeze(
                0
            )
            print(
                "Warning: No dynamics information for link: {}, setting all inertial properties to 1.".format(
                    link.name
                )
            )

        return body_params


if __name__ == '__main__':
    hand = MJCFRobotModel(get_mjcf_path() / 'shadow_hand_series_e.xml')
    print(hand.find_joint_of_body('forearm'))
    print(hand.get_body_parameters(0, hand._bodies['thmiddle']))
