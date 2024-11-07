"""
MIT License

Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List
from enum import Enum
import concurrent.futures

from mmd.common.experiments import TrialSuccessStatus
# Project imports.
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes
from mmd.common.conflicts import VertexConflict, Conflict
from mmd.common.constraints import MultiPointConstraint
from mmd.common.experiences import PathExperience, PathBatchExperience
from mmd.common.pretty_print import *
from mmd.common import densify_trajs, smooth_trajs, is_multi_agent_start_goal_states_valid, global_pad_paths
from mmd.config import MMDParams as params
from mmd.planners.multi_agent.cbs import SearchState  # Holding multi-agent paths and constraints information.


class PrioritizedPlanning:
    """
    Prioritized Planning (PP) algorithm.
    """
    def __init__(self, low_level_planner_l,
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 start_time_l: List[int] = None,
                 reference_robot=None,
                 reference_task=None,
                 **kwargs):

        # Some parameters:
        self.low_level_choose_path_from_batch_strategy = params.low_level_choose_path_from_batch_strategy
        # Set the low level planners.
        self.low_level_planner_l = low_level_planner_l
        self.num_agents = len(start_l)
        self.agent_color_l = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))
        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        if start_time_l is None:
            start_time_l = [0] * self.num_agents
        else:
            self.start_time_l = start_time_l
        # Keep a reference robot for collision checking in a group of robots.
        if reference_robot is None:
            print(CYAN + 'Using the first robot in the low level planner list as the reference robot.' + RESET)
            self.reference_robot = self.low_level_planner_l[0].robot
        else:
            self.reference_robot = reference_robot
        if reference_task is None:
            print(CYAN + 'Using the first task in the low level planner list as the reference task.' + RESET)
            self.reference_task = self.low_level_planner_l[0].task
        else:
            self.reference_task = reference_task
        self.tensor_args = self.low_level_planner_l[0].tensor_args
        self.results_dir = self.low_level_planner_l[0].results_dir
        # Check for collisions between robots, and between robots and obstacles, in their start and goal states.
        if not is_multi_agent_start_goal_states_valid(self.reference_robot,
                                                      self.reference_task,
                                                      self.start_state_pos_l,
                                                      self.goal_state_pos_l):
            raise ValueError('Start or goal states are invalid.')

    def render_paths(self, paths_l: List[torch.Tensor], constraints_l: List[MultiPointConstraint] = None,
                     animation_duration: float = 10.0, output_fpath=None, n_frames=None, plot_trajs=True,
                     show_robot_in_image=True):
        # Render
        planner_visualizer = PlanningVisualizer(
            task=self.reference_task,
        )

        # Add batch dimension to all paths.
        paths_l = [path.unsqueeze(0) for path in paths_l]

        # If animation_duration is None or 0, don't animate and save an image instead.
        if animation_duration is None or animation_duration == 0:
            fig, ax = create_fig_and_axes()
            for agent_id in range(self.num_agents):
                planner_visualizer.render_robot_trajectories(
                    fig=fig,
                    ax=ax,
                    trajs=paths_l[agent_id],
                    start_state=self.start_state_pos_l[agent_id],
                    goal_state=self.goal_state_pos_l[agent_id],
                    colors=[self.agent_color_l[agent_id]],
                    show_robot_in_image=show_robot_in_image
                )
            if output_fpath is None:
                output_fpath = os.path.join(self.results_dir, 'robot-traj.png')
            if not output_fpath.endswith('.png'):
                output_fpath = output_fpath + '.png'
            print(f'Saving image to: file://{os.path.abspath(output_fpath)}')
            plt.axis('off')
            plt.savefig(output_fpath, dpi=100, bbox_inches='tight', pad_inches=0)
            return

        base_file_name = Path(os.path.basename(__file__)).stem
        if output_fpath is None:
            output_fpath = os.path.join(self.results_dir, f'{base_file_name}-robot-traj.gif')
        # Render the paths.
        print(f'Rendering paths and saving to: file://{os.path.abspath(output_fpath)}')
        planner_visualizer.animate_multi_robot_trajectories(
            trajs_l=paths_l,
            start_state_l=self.start_state_pos_l,
            goal_state_l=self.goal_state_pos_l,
            plot_trajs=plot_trajs,
            video_filepath=output_fpath,
            n_frames=max((2, paths_l[0].shape[1])) if n_frames is None else n_frames,
            # n_frames=pos_trajs_iters[-1].shape[1],
            anim_time=animation_duration,
            constraints=constraints_l,
            colors=self.agent_color_l
        )

    def plan(self, runtime_limit=1000):
        """
        Plan a path from start to goal. Do it for one agent at a time.
        """
        startt = time.time()
        success_status = TrialSuccessStatus.UNKNOWN

        # Empty root node.
        root = SearchState([], [])
        for i in range(len(self.low_level_planner_l)):
            constraint_l = self.create_soft_constraints_from_other_agents_paths(root, agent_id=i)
            for c in constraint_l:
                # Make the constraints hard. This makes their collection act as a priority constraint.
                c.is_soft = False
                # Clip to range.
                c.t_range_l = [(max(0, min(t_range[0], params.horizon - 1)),
                                min(params.horizon - 1, t_range[1]))
                               for t_range in c.t_range_l]

            planner_output = self.low_level_planner_l[i](self.start_state_pos_l[i],
                                                         self.goal_state_pos_l[i],
                                                         constraints_l=constraint_l)
            if planner_output.trajs_final_free_idxs.shape[0] == 0:
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                break

            ix_best_traj = planner_output.idx_best_traj
            # Update the root node.
            root.path_bl.append(planner_output.trajs_final)
            root.ix_best_path_in_batch_l.append(ix_best_traj)
            root.conflict_l = self.get_conflicts(root)
            # Check if this agent i should use another path that has fewer conflicts.
            for ix_traj in planner_output.trajs_final_free_idxs:
                temp_state = root.get_copy()
                temp_state.ix_best_path_in_batch_l[i] = ix_traj
                conflict_l = self.get_conflicts(temp_state)
                if root.conflict_l is None:
                    root.ix_best_path_in_batch_l[i] = ix_traj
                    root.conflict_l = conflict_l
                if len(conflict_l) < len(root.conflict_l):
                    print(f'Found a path with fewer conflicts: {len(conflict_l)} < {len(root.conflict_l)}')
                    root.ix_best_path_in_batch_l[i] = ix_traj
                    root.conflict_l = conflict_l

            # Check if the runtime limit has been reached.
            if time.time() - startt > runtime_limit:
                print('Runtime limit reached.')
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                break

        # Extract the best path from the batch.
        best_path_l = [root.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
                       enumerate(root.ix_best_path_in_batch_l)]
        # Check for conflicts.
        conflict_l = self.get_conflicts(root)
        print(RED + 'Conflicts root node:', len(conflict_l), RESET)
        if success_status == TrialSuccessStatus.UNKNOWN:
            if len(conflict_l) > 0:
                success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS
            else:
                success_status = TrialSuccessStatus.SUCCESS

        # Global pad before returning.
        best_path_l = global_pad_paths(best_path_l, self.start_time_l)
        # Return the best path, the CT nodes expanded (0), success status, and num collisions in output.
        return best_path_l, 0, success_status, len(conflict_l)

    def create_soft_constraints_from_other_agents_paths(self, state: SearchState, agent_id: int) -> List[MultiPointConstraint]:
        """
        Create soft constraints from the paths of other agents.
        """
        if len(state.path_bl) == 0:
            return []

        agent_constraint_l = []  # The output list of soft constraints.
        q_l = []
        t_range_l = []
        radius_l = []
        num_agents_in_state = len(state.path_bl)
        for agent_id_other in range(num_agents_in_state):
            if agent_id_other != agent_id:
                best_path_other_agent = \
                    state.path_bl[agent_id_other][state.ix_best_path_in_batch_l[agent_id_other]].squeeze(0)
                best_path_pos_other_agent = self.reference_robot.get_position(best_path_other_agent)
                for t_other_agent in range(0, len(best_path_other_agent), 1):
                    t_agent = t_other_agent + self.start_time_l[agent_id_other] - self.start_time_l[agent_id]
                    # The last timestep index for this agent is the lenfth of its path - 1.
                    # If it does not have a path stored, then create constraints for all timesteps
                    # in the path of the other agent (starting from zero).
                    T_agent = len(state.path_bl[agent_id_other][0]) - 1
                    if agent_id >= len(state.path_bl):
                        T_agent = len(best_path_other_agent) - 1
                    else:
                        T_agent = len(state.path_bl[agent_id][0]) - 1

                    if 1 <= t_agent <= T_agent:
                        q_l.append(best_path_pos_other_agent[t_other_agent])
                        t_range_l.append((t_agent, t_agent + 1))
                        radius_l.append(params.vertex_constraint_radius)

        if len(q_l) > 0:
            soft_constraint = MultiPointConstraint(q_l=q_l, t_range_l=t_range_l)
            soft_constraint.radius_l = radius_l
            soft_constraint.is_soft = True
            agent_constraint_l.append(soft_constraint)
        return agent_constraint_l


    def get_conflicts(self, state: SearchState) -> List[Conflict]:
        """
        Find conflicts between paths.
        """
        # Get a list of best paths from the search state. Each path is shape (H, q_dim).
        best_path_l = [state.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
                       enumerate(state.ix_best_path_in_batch_l)]
        # Pad the paths to make them all the same length. Different agents may have different start times, accommodate that.
        best_path_l = global_pad_paths(best_path_l, self.start_time_l)

        # Get the positions of all robots at all times.
        paths_pos_l = [self.reference_robot.get_position(path) for path in best_path_l]
        # Return empty if no paths.
        if len(paths_pos_l) == 0:
            return []
        max_t = max([len(path) for path in paths_pos_l])  # This should be the same for all agents.
        # Find conflicts.
        conflicts = []
        densification_factor = 1
        paths_pos_l_dense = densify_trajs(paths_pos_l, densification_factor)
        for t in range(max_t):
            # Check all robot positions at [t * densification_factor, (t + 1) * densification_factor].
            for t_dense in range(t * densification_factor, (t + 1) * densification_factor):
                # Note(yorai): the last iteration will likely only have the last state in the path repeated. This is ok.
                positions_dense = [path[t_dense] if len(path) > t_dense else path[-1] for path in paths_pos_l_dense]
                positions_dense = torch.stack(positions_dense)  # ..., n_robots, q_dim
                # Find conflicts.
                collisions_pairwise, _ = self.reference_robot.check_rr_collisions(positions_dense)
                if torch.any(collisions_pairwise):
                    # This node may only have paths from a (contiguous) subset of agents.
                    num_agents = positions_dense.shape[0]
                    for agent_id_a in range(num_agents):
                        for agent_id_b in range(agent_id_a + 1, num_agents):
                            if collisions_pairwise[..., agent_id_a, agent_id_b]:
                                t_a = t if len(paths_pos_l[agent_id_a]) > t else len(paths_pos_l[agent_id_a]) - 1
                                t_a_dense = t_dense if len(paths_pos_l_dense[agent_id_a]) > t_dense else len(paths_pos_l_dense[agent_id_a]) - 1
                                t_b = t if len(paths_pos_l[agent_id_b]) > t else len(paths_pos_l[agent_id_b]) - 1
                                t_b_dense = t_dense if len(paths_pos_l_dense[agent_id_b]) > t_dense else len(paths_pos_l_dense[agent_id_b]) - 1
                                # We set the conflict to happen at the beginning of the interpolated time interval,
                                # but use the interpolated states.
                                conflicts.append(
                                    VertexConflict([agent_id_a, agent_id_b],
                                                   # [paths_pos_l[agent_id_a][t_a], paths_pos_l[agent_id_b][t_b]],
                                                   [
                                                       paths_pos_l_dense[agent_id_a][t_a_dense],
                                                       paths_pos_l_dense[agent_id_b][t_b_dense]
                                                   ],
                                                   t))
                                # print(f'Conflict between agents {agent_id_a} and {agent_id_b} at time {t}.')
        return conflicts
