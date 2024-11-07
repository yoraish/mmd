import torch

from .tasks import *
from torch_robotics.environments import *
from typing import Dict
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from mmd.common.pretty_print import *

class TaskEnsemble(Task):
    def __init__(self,
                 tasks: Dict[int, Task],
                 transforms: Dict[int, torch.Tensor],
                 **kwargs):
        self.tasks = tasks  # Dict[int, PlanningTask]
        self.transforms = transforms  # Dict[int, torch.Tensor [dx dy] in 2D.]
        envs_dict = {k: task.env for k, task in tasks.items()}
        env = EnvEnsemble(envs_dict, transforms)
        super().__init__(env, tasks[0].robot, **kwargs)

    def transform_q(self, task_id: int, q):
        if q.shape[-1] > self.transforms[task_id].shape[0]:
            transform = torch.cat(
                [self.transforms[task_id], torch.zeros(q.shape[-1] - self.transforms[task_id].shape[0],
                                                       device=self.transforms[task_id].device)])
        else:
            transform = self.transforms[task_id]
        return q + transform

    def inverse_transform_q(self, task_id: int, q):
        if q.shape[-1] > self.transforms[task_id].shape[0]:
            transform = torch.cat(
                [self.transforms[task_id], torch.zeros(q.shape[-1] - self.transforms[task_id].shape[0],
                                                       device=self.transforms[task_id].device)])
        else:
            transform = self.transforms[task_id]
        return q - transform


class PlanningTaskEnsemble(TaskEnsemble):
    def __init__(self,
                 tasks: Dict[int, PlanningTask],
                 transforms: Dict[int, torch.Tensor],
                 ws_limits=None,
                 **kwargs):
        super().__init__(tasks, transforms, **kwargs)
        self.ws_limits = self.env.limits if ws_limits is None else ws_limits
        self.ws_min = self.ws_limits[0]
        self.ws_max = self.ws_limits[1]

    def get_collision_fields(self, task_id=None):
        if task_id is None:
            tot_fields = []
            for task in self.tasks.values():
                col_fields = task.get_collision_fields()
                tot_fields.extend(col_fields)
            return tot_fields
        return self.tasks[task_id].get_collision_fields()

    def get_collision_fields_extra_objects(self, task_id=None):
        if task_id is None:
            tot_fields = []
            for task in self.tasks.values():
                col_fields = task.get_collision_fields_extra_objects()
                tot_fields.extend(col_fields)
            return tot_fields
        return self.tasks[task_id].get_collision_fields_extra_objects()

    def distance_q(self, task_id: int, q1, q2):
        return self.tasks[task_id].distance_q(q1, q2)

    def sample_q(self, task_id: int, without_collision=True, **kwargs):
        return self.tasks[task_id].sample_q(without_collision=without_collision, **kwargs)

    def random_coll_free_q(self, task_id: int, n_samples=1, max_samples=1000, max_tries=1000):
        return self.tasks[task_id].random_coll_free_q(n_samples, max_samples, max_tries)

    def get_traj_unnormalized(self, task_id: int,
                              datasets,
                              traj_normalized):
        trajs_iters = datasets[task_id].unnormalize_trajectories(traj_normalized)
        trajs_final = trajs_iters[-1]
        trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = \
            (self.tasks[task_id].get_trajs_collision_and_free(trajs_final, return_indices=True))
        return trajs_iters, trajs_final, trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs

    def get_stats(self,
                  task_id: int,
                  trajs_iters,
                  trajs_final,
                  trajs_final_coll,
                  trajs_final_coll_idxs,
                  trajs_final_free,
                  trajs_final_free_idxs,
                  t_total,
                  save_data=True,
                  results_dir=None
                  ):
        success_free_trajs = self.tasks[task_id].compute_success_free_trajs(trajs_final)
        fraction_free_trajs = self.tasks[task_id].compute_fraction_free_trajs(trajs_final)
        collision_intensity_trajs = self.tasks[task_id].compute_collision_intensity_trajs(trajs_final)

        print(f'success: {success_free_trajs}')
        print(f'percentage free trajs: {fraction_free_trajs * 100:.2f}')
        print(f'percentage collision intensity: {collision_intensity_trajs * 100:.2f}')

        # compute costs only on collision-free trajectories
        traj_final_free_best = None
        idx_best_traj = None
        cost_best_free_traj = None
        cost_smoothness = None
        cost_path_length = None
        cost_all = None
        variance_waypoint_trajs_final_free = None
        if trajs_final_free is not None:
            cost_smoothness = compute_smoothness(trajs_final_free, self.tasks[task_id].robot)
            print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

            cost_path_length = compute_path_length(trajs_final_free, self.tasks[task_id].robot)
            print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

            # compute best trajectory
            cost_all = cost_path_length + cost_smoothness
            idx_best_traj_free = torch.argmin(cost_all).item()
            traj_final_free_best = trajs_final_free[idx_best_traj_free]
            cost_best_free_traj = torch.min(cost_all).item()
            print(f'cost best: {cost_best_free_traj:.3f}')

            # variance of waypoints
            variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, self.tasks[task_id].robot)
            print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')

        print(f'\n--------------------------------------\n')

        ########################################################################################################################
        # Save data
        results_data_dict = {
            'trajs_iters': trajs_iters,
            'trajs_final_coll': trajs_final_coll,
            'trajs_final_coll_idxs': trajs_final_coll_idxs,
            'trajs_final_free': trajs_final_free,
            'trajs_final_free_idxs': trajs_final_free_idxs,
            'success_free_trajs': success_free_trajs,
            'fraction_free_trajs': fraction_free_trajs,
            'collision_intensity_trajs': collision_intensity_trajs,
            # 'idx_best_traj_free': idx_best_traj_free,
            'traj_final_free_best': traj_final_free_best,
            'cost_best_free_traj': cost_best_free_traj,
            'cost_path_length_trajs_final_free': cost_path_length,
            'cost_smoothness_trajs_final_free': cost_smoothness,
            'cost_all_trajs_final_free': cost_all,
            'variance_waypoint_trajs_final_free': variance_waypoint_trajs_final_free,
            't_total': t_total
        }

        if save_data:
            import os
            import pickle
            with open(os.path.join(results_dir, f'results_data_dict_model{task_id}.pickle'), 'wb') as handle:
                pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return results_data_dict

    def combine_trajs(self,
                      results_ensemble: Dict[int, Dict]):
        keys = list(results_ensemble.values())[0].keys()
        for task_id, results in results_ensemble.items():
            for k in keys:
                if "idxs" not in k and k.startswith("traj"):
                    if results[k] is not None:
                        results_ensemble[task_id][k] = self.transform_q(task_id, results[k])
                    else:
                        results_ensemble[task_id][k] = None

        results = {'trajs_iters': torch.cat([results_ensemble[task_id]['trajs_iters'] for task_id in
                                             results_ensemble.keys()], dim=-2)}
        trajs_final = results['trajs_iters'][-1]
        coll_ids = torch.tensor([], **self.tensor_args)
        for batch_id in range(results_ensemble[0]['trajs_iters'].shape[1]):
            for task_id in results_ensemble.keys():
                if batch_id in results_ensemble[task_id]['trajs_final_coll_idxs']:
                    coll_ids = torch.cat((coll_ids, torch.tensor([batch_id], **self.tensor_args)))
                    break
        free_ids = torch.tensor([i for i in range(results_ensemble[0]['trajs_iters'].shape[1]) if i not in coll_ids],
                                **self.tensor_args)

        results['trajs_final_coll_idxs'] = coll_ids.long()
        results['trajs_final_free_idxs'] = free_ids.long()
        results['trajs_final_coll'] = trajs_final[:, results['trajs_final_coll_idxs'], ...] \
            if len(coll_ids) > 0 else torch.tensor([], **self.tensor_args)
        results['trajs_final_free'] = trajs_final[results['trajs_final_free_idxs'], :, ...] \
            if len(free_ids) > 0 else torch.tensor([], **self.tensor_args)
        results['success_free_trajs'] = 1 if len(free_ids) > 0 else 0
        results['fraction_free_trajs'] = len(free_ids) / results['trajs_iters'].shape[1]
        results['collision_intensity_trajs'] = 1 - results['fraction_free_trajs']
        if len(free_ids) > 0:
            # results['cost_smoothness_trajs_final_free'] = torch.sum(
            #     torch.cat([results_ensemble[task_id]['cost_smoothness_trajs_final_free'].unsqueeze(0)
            #                  for task_id in results_ensemble.keys()],
            #                 dim=-1), dim=-1)
            # results['cost_path_length_trajs_final_free'] = torch.sum(
            #     torch.cat([results_ensemble[task_id]['cost_path_length_trajs_final_free'].unsqueeze(0)
            #                  for task_id in results_ensemble.keys()],
            #                 dim=-1), dim=-1)
            results['cost_smoothness_trajs_final_free'] = compute_smoothness(results['trajs_final_free'], self.robot)
            results['cost_path_length_trajs_final_free'] = compute_path_length(results['trajs_final_free'], self.robot)

            # results['cost_all_trajs_final_free'] = torch.sum(
            #     torch.cat([results_ensemble[task_id]['cost_all_trajs_final_free'].unsqueeze(0)
            #                  for task_id in results_ensemble.keys()],
            #                 dim=-1), dim=-1)

            results['cost_all_trajs_final_free'] = results['cost_smoothness_trajs_final_free'] + results['cost_path_length_trajs_final_free']

            # results['variance_waypoint_trajs_final_free'] = torch.sum(
            #     torch.cat([results_ensemble[task_id]['variance_waypoint_trajs_final_free'].unsqueeze(0)
            #                  for task_id in results_ensemble.keys()],
            #                 dim=-1), dim=-1)
            results['variance_waypoint_trajs_final_free'] = compute_variance_waypoints(results['trajs_final_free'], self.robot)

            idx_best_traj_among_free_trajs = torch.argmin(results['cost_all_trajs_final_free'], dim=-1)
            # Index of best free trajectory among all trajectories.
            results['idx_best_traj'] = results['trajs_final_free_idxs'][idx_best_traj_among_free_trajs]
            results['traj_final_free_best'] = results['trajs_final_free'][idx_best_traj_among_free_trajs, ...]
            results['cost_best_free_traj'] = torch.min(results['cost_all_trajs_final_free'], dim=-1)[0]
        results['t_total'] = results_ensemble[0]['t_total']
        return results

        # return results_ensemble

    def compute_collision(self, x, **kwargs):
        q_pos = self.robot.get_position(x)
        return self._compute_collision_or_cost(q_pos, field_type='occupancy', **kwargs)

    def compute_collision_cost(self, x, **kwargs):
        q_pos = self.robot.get_position(x)
        return self._compute_collision_or_cost(q_pos, field_type='sdf', **kwargs)

    def _compute_collision_or_cost(self, q, field_type='occupancy', **kwargs):
        # q.shape needs to be reshaped to (batch, horizon, q_dim)
        q_original_shape = q.shape
        b = 1
        h = 1
        collisions = None
        if q.ndim == 1:
            q = q.unsqueeze(0).unsqueeze(0)  # add batch and horizon dimensions for interface
            collisions = torch.ones((1,), **self.tensor_args)
        elif q.ndim == 2:
            b = q.shape[0]
            q = q.unsqueeze(1)  # add horizon dimension for interface
            collisions = torch.ones((b, 1), **self.tensor_args)  # (batch, 1)
        elif q.ndim == 3:
            b = q.shape[0]
            h = q.shape[1]
            collisions = torch.ones((b, h), **self.tensor_args)  # (batch, horizon)
        elif q.ndim > 3:
            raise NotImplementedError
        collisions = collisions.long()
        # get the right task based on q values and each task limits
        task_ids = self.infer_task_id_from_q(q)  # Shape (num_qs, ).
        for task_id in self.tasks.keys():
            mask = task_ids == task_id
            if torch.any(mask):
                q_task = q[mask]
                q_task = self.inverse_transform_q(task_id, q_task)
                mask = mask.view(-1)
                collisions[mask] = self.tasks[task_id]._compute_collision_or_cost(q_task, field_type=field_type, **kwargs)

        return collisions.view(q_original_shape[:-1])

        # return self.tasks[task_id]._compute_collision_or_cost(self.inverse_transform_q(task_id, q), field_type=field_type, **kwargs)

    def get_trajs_collision_and_free(self, trajs, return_indices=False, num_interpolation=5):
        # TEST TEST TEST. Return all valid.
        # TODO(yorai): get_trajs_collision_and_free returns all free. This is for visualization, but still needs to be
        #  fixed.
        if return_indices:
            return None, torch.tensor([]), trajs, torch.tensor([i for i in range(trajs.shape[0])]), torch.tensor([False for i in range(trajs.shape[1])])
        return None, trajs

        assert trajs.ndim == 3 or trajs.ndim == 4
        N = 1
        if trajs.ndim == 4:  # n_goals (or steps), batch of trajectories, length, dim
            N, B, H, D = trajs.shape
            trajs_new = einops.rearrange(trajs, 'N B H D -> (N B) H D')
        else:
            B, H, D = trajs.shape
            trajs_new = trajs

        # ##############################################################################################################
        # # compute collisions on a finer interpolated trajectory
        # trajs_interpolated = interpolate_traj_via_points(trajs_new, num_interpolation=num_interpolation)
        trajs_interpolated_all = []
        trajs_waypoints_collisions_all = {}
        for task_id, task in self.tasks.items():
            ###############################################################################################################
            # compute collisions on a finer interpolated trajectory
            trajs_interpolated = interpolate_traj_via_points(
                                        trajs_new[:, task_id * params.horizon:(task_id + 1) * params.horizon, ...],
                                        num_interpolation=num_interpolation)
            trajs_interpolated_all.append(trajs_interpolated)
            # TODO(yoraish): this is a hack to accomodate the planar disk robot which only has one point of collision
            #  but a volume nonetheless. If robot has a radius, then use it as margin.
            if getattr(self.robot, 'radius', None) is not None:
                margin = self.robot.radius
            trajs_waypoints_collisions = task.compute_collision(trajs_interpolated, margin=margin)
            trajs_waypoints_collisions_all[task_id] = trajs_waypoints_collisions

        trajs = torch.cat(trajs_interpolated_all, dim=1)
        trajs_waypoints_collisions = torch.stack(list(trajs_waypoints_collisions_all.values()), dim=0).all(dim=0)
        if trajs.ndim == 4:
            trajs_waypoints_collisions = einops.rearrange(trajs_waypoints_collisions, '(N B) H -> N B H', N=N, B=B)

        trajs_free_idxs = torch.argwhere(torch.logical_not(trajs_waypoints_collisions).all(dim=-1))
        trajs_coll_idxs = torch.argwhere(trajs_waypoints_collisions.any(dim=-1))

        # Return trajectories free and in collision
        if trajs.ndim == 4:
            trajs_free = trajs[trajs_free_idxs[:, 0], trajs_free_idxs[:, 1], ...]
            if trajs_free.ndim == 2:
                trajs_free = trajs_free.unsqueeze(0).unsqueeze(0)
            trajs_coll = trajs[trajs_coll_idxs[:, 0], trajs_coll_idxs[:, 1], ...]
            if trajs_coll.ndim == 2:
                trajs_coll = trajs_coll.unsqueeze(0).unsqueeze(0)
        else:
            trajs_free = trajs[trajs_free_idxs.squeeze(), ...]
            if trajs_free.ndim == 2:
                trajs_free = trajs_free.unsqueeze(0)
            trajs_coll = trajs[trajs_coll_idxs.squeeze(), ...]
            if trajs_coll.ndim == 2:
                trajs_coll = trajs_coll.unsqueeze(0)

        if trajs_coll.nelement() == 0:
            trajs_coll = None
        if trajs_free.nelement() == 0:
            trajs_free = None

        if return_indices:
            return trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_collisions
        return trajs_coll, trajs_free

    def infer_task_id_from_q_idx(self, q_idx):
        task_id = int(q_idx // params.horizon)
        task = self.tasks[task_id]
        return task_id, task

    def infer_task_id_from_q(self, q):
        """
        Infer the task id from the joint values.
        :param q: Shape (B, H, q_dim).
        :return: task_ids: Shape (B, ). With each entry is the task id of the corresponding joint values traj.
        """
        q_pos = self.robot.get_position(q)
        # check the limits of each task
        env_limits = []
        for task_id, task in self.tasks.items():
            limits = task.env.limits
            # transform the limits
            lower = self.transform_q(task_id, limits[0])
            upper = self.transform_q(task_id, limits[1])
            env_limits.append((lower, upper))
        task_ids = torch.zeros(q_pos.shape[0], device=q_pos.device, dtype=torch.long) - 1

        for i, limits in enumerate(env_limits):
            # if torch.all(q_pos >= limits[0]) and torch.all(q_pos <= limits[1]):
            #     task_id = i
            #     break
            mask = torch.logical_and(q_pos >= limits[0], q_pos <= limits[1])
            mask = mask.all(dim=-1).squeeze()
            task_ids[mask] = i

        return task_ids  # Shape (num_qs, ).
