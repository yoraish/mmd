from copy import deepcopy
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph
import json
import torch
import numpy as np
import torch.distributions as D
from torch_robotics.torch_kinematics_tree.geometrics.utils import MinMaxScaler, euclidean_distance


class Skeleton():

    def __init__(self, data, node_list=None, distance_func=None, device='cpu'):
        self.load_skeleton(data)
        self.node_pos = nx.get_node_attributes(self.skeleton, 'pose')
        self.node_var = nx.get_node_attributes(self.skeleton, 'var')
        self.node_name = nx.get_node_attributes(self.skeleton, 'name')
        self.node_ids = list(self.node_name.keys())
        if node_list is None:
            node_list = deepcopy(self.node_ids)
        self.node_list = node_list
        self.num_node = len(self.node_ids)
        # self.node_ids.sort()
        if distance_func is None:
            distance_func = euclidean_distance
        self.distance_func = distance_func
        self.device = device
        self.node_dist = {
            id: D.MultivariateNormal(torch.tensor(self.node_pos[id]).to(device),
                                     covariance_matrix=self.node_var[id]  * torch.eye(2, device=device))
            for id in self.node_ids
        }
        # compute link lengths
        self.compute_link_lengths()
        # for plotting
        self.three_d = len(self.node_pos[self.node_list[-1]]) == 3

    def load_skeleton(self, data):
        if isinstance(data, Path):
            with open(data) as f:
                data = json.load(f)
                self.skeleton = json_graph.node_link_graph(data)
        else:  # assume dict
            self.skeleton = json_graph.node_link_graph(data)
    
    def compute_link_lengths(self):
        for u, v in self.skeleton.edges():
            u_pos, v_pos = torch.tensor(self.node_pos[u], device=self.device), torch.tensor(self.node_pos[v], device=self.device)
            self.skeleton[u][v][0]['length'] = self.distance_func(u_pos, v_pos).squeeze().cpu().item()

    def compute_adjacency_matrix(self, weight='length'):
        A = nx.adjacency_matrix(self.skeleton, weight=weight)
        A = A.todense()
        return A

    def get_edges(self, weighted=False):
        edges = np.array(list(self.skeleton.edges()))
        if weighted:
            weights = [[self.skeleton[u][v][0]['length']] for u, v in edges]
        else:
            weights = [[1.0]] * len(edges)
        weights = np.array(weights)
        return edges, weights

    def update_node_pose(self, node_dict):
        nx.set_node_attributes(self.skeleton, node_dict)
        self.node_pos = nx.get_node_attributes(self.skeleton, 'pose')
        self.compute_link_lengths()

    def compute_self_distance(self, length_cutoff=None, normalized=False):
        res = dict(nx.all_pairs_dijkstra_path_length(self.skeleton, cutoff=length_cutoff, weight='length'))
        # since res is a generator
        D = np.zeros((len(self.node_list), len(self.node_list)))
        for i, id1 in enumerate(self.node_list):
            for j, id2 in enumerate(self.node_list):
                D[i, j] = res[id1][id2]
        if normalized:
            scaler = MinMaxScaler()
            D = scaler.scale(D)
        return D

    def get_all_neighbors(self):
        N = np.zeros(len(self.node_list))
        for i, id in enumerate(self.node_list):
            N[i] = len(list(self.skeleton.neighbors(id)))
        return N

    def get_pos_tensor(self):
        return torch.tensor(np.array([self.node_pos[id] for id in self.node_ids]), device=self.device)
    
    def get_pos_tensor_from_list(self, node_list=None):
        if node_list is None:
            node_list = self.node_list
        return torch.tensor(np.array([self.node_pos[id] for id in node_list]), device=self.device)

    def get_node_list_dict(self):
        return {k: v for k, v in self.node_pos.items() if k in self.node_list}

    def draw_skeleton(self, pos=None, shift=None, with_labels=False, color='b', ax=None):
        if pos is None:
            pos = self.node_pos
        else:
            if torch.is_tensor(next(iter(pos.values()))):
                pos = {k: v.cpu().numpy() for k, v in pos.items()}
        if shift is not None:
            pos = {k: v + shift for k, v in pos.items()}
        if self.three_d:
            ax.scatter3D(
                xs=[pos[id][0] for id in self.node_list],
                ys=[pos[id][1] for id in self.node_list],
                zs=[pos[id][2] for id in self.node_list],
                color=color,
                linewidth=10)
            for u, v in self.skeleton.edges():
                if u in self.node_list and v in self.node_list:
                    ax.plot3D(
                        xs=[pos[u][0], pos[v][0]],
                        ys=[pos[u][1], pos[v][1]],
                        zs=[pos[u][2], pos[v][2]],
                        color='k',
                        linewidth=7)
        else:
            nx.draw(self.skeleton, pos, ax, labels=self.node_name, with_labels=with_labels, font_size=6)

    def sample_posture(self, batch_size):
        postures = []
        for id in self.node_ids:
            postures.append(self.node_dist[id].sample((batch_size, )))
        return torch.stack(postures, dim=1)


def get_skeleton_from_model(model, q_calib, link_list=None):
    '''Get Skeleton from robots model'''
    if isinstance(q_calib, np.ndarray) or isinstance(q_calib, list):
        q_calib = torch.tensor(q_calib, device=model._device)
    if q_calib.ndim == 1:
        q_calib = q_calib.unsqueeze(0)
    H = model.compute_forward_kinematics_all_links(q_calib, return_dict=True)
    node_list = []
    # only get euclidean positions
    for name, frame in H.items():
        pose = frame.translation
        pose = pose.squeeze().detach().cpu().numpy()
        node_list.append({
            'id': name, 'name': name, 'pose': pose, 'var': 0.001
        })
    child_map = model._model.robot.child_map
    edge_list = []
    for parent, childs in child_map.items():
        for child in childs:
            edge_list.append({
                'source': parent,
                'target': child[1]
            })
    data = {'nodes': node_list, 'links': edge_list}
    skeleton = Skeleton(data, node_list=link_list, device=model._device)
    return skeleton


def get_skeleton_from_mediapipe(landmark_list, connections, link_list=None, present_thres=0.5, vis_thres=0.5, mirror=False, relative_pose=False, shift=np.zeros(3), device='cpu'):
    if landmark_list is None:
        return None

    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < vis_thres) or
            (landmark.HasField('presence') and
            landmark.presence < present_thres)):
            continue
        if mirror:
            plotted_landmarks[idx] = [-landmark.z, -landmark.x, landmark.y]
        else:
            plotted_landmarks[idx] = [landmark.z, -landmark.x, -landmark.y]


    if link_list is not None:
        # return None if fail to recognize correct links
        if not all(item in plotted_landmarks for item in link_list):
            return None
    node_list = []
    # only get euclidean positions
    base_pose = np.array(plotted_landmarks[0])
    for name, pose in plotted_landmarks.items():
        pose = np.array(pose) 
        node_list.append({
            'id': name, 'name': name, 'pose': ((pose - base_pose) if relative_pose else pose) + shift, 'var': 0.001
        })
    num_landmarks = len(landmark_list.landmark)
    edge_list = []
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
            raise ValueError(f'Landmark index is out of range. Invalid connection '
                             f'from landmark #{start_idx} to landmark #{end_idx}.')
        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            edge_list.append({
                'source': start_idx,
                'target': end_idx
            })
    data = {'nodes': node_list, 'links': edge_list}
    skeleton = Skeleton(data, node_list=link_list, device=device)
    return skeleton
