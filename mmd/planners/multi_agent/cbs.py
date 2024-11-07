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
from math import floor, ceil
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Dict, Type
from enum import Enum
import concurrent.futures

# Project imports.
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes
from mmd.common.conflicts import VertexConflict, Conflict, PointConflict, EdgeConflict
from mmd.common.constraints import MultiPointConstraint, Constraint
from mmd.common.conflict_conversion import convert_conflicts_to_constraints
from mmd.common.experiences import PathExperience, PathBatchExperience
from mmd.common.pretty_print import *
from mmd.common import densify_trajs, smooth_trajs, is_multi_agent_start_goal_states_valid, global_pad_paths
from mmd.config import MMDParams as params
from mmd.common.experiments import TrialSuccessStatus
from mmd.planners.single_agent import MPD

"""
Some comments:
1. This assumes, as of now, a homogeneous team of robots (so that asking for one to compute
    the conflicts with others makes sense with a robot method).
"""


class CBSExperienceReuseStrategy(Enum):
    """
    Enum for replanning strategies in CBS.
    """
    NONE = 0
    XCBS = 1
    NOISE_AS_EXPERIENCE = 2


class SearchState:
    """
    Constraint Tree node for CBS.
    """

    def __init__(self, ix_best_path_in_batch_l, path_bl, constraints={}):
        self.path_bl = path_bl  # List of batch of paths. (list of n_agents) x B x H x q_dim.
        self.ix_best_path_in_batch_l = ix_best_path_in_batch_l  # List of indices of the best path in the batch. (n_agents,).
        self.conflict_l = []
        self.constraints = constraints  # Map of agent_id: List[Constraint].
        self.g = float('inf')  # Cost to reach this node.

    def update_g_l2(self):
        """
        Update the cost to reach this node.
        """
        self.g = 0
        for i, ix_best_path_in_batch in enumerate(self.ix_best_path_in_batch_l):
            path = self.path_bl[i][ix_best_path_in_batch]
            path_cost = torch.norm(path[1:] - path[:-1], dim=-1).sum()
            self.g += path_cost

    def add_constraint(self, agent_id, constraint):
        """
        Add a constraint to the state.
        """
        if agent_id not in self.constraints:
            self.constraints[agent_id] = []
        print(RED + f'Adding constraint for agent {agent_id}. Before:', len(self.constraints[agent_id]), RESET)
        self.constraints[agent_id].append(constraint)
        print(GREEN + f'After:', len(self.constraints[agent_id]), RESET)

    def get_copy(self):
        """
        Create a copy of the state.
        """
        new_ix_best_path_in_batch_l = self.ix_best_path_in_batch_l.copy()
        new_path_bl = [path_b.clone() for path_b in self.path_bl]
        new_constraints = {k: [c.get_copy() for c in v] for k, v in self.constraints.items()}
        new_state = SearchState(new_ix_best_path_in_batch_l, new_path_bl, new_constraints)
        new_state.conflict_l = self.conflict_l
        new_state.g = self.g
        return new_state


class CBS:
    """
    Conflict-Based Search (CBS) algorithm.
    """
    def __init__(self, low_level_planner_l,
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 start_time_l: List[int] = None,
                 is_xcbs=False,
                 is_ecbs=True,
                 conflict_type_to_constraint_types: Dict[Type[Conflict], Type[Constraint]] = None,
                 reference_robot=None,
                 reference_task=None,
                 **kwargs):
        # Some parameters:
        self.low_level_choose_path_from_batch_strategy = params.low_level_choose_path_from_batch_strategy
        # Set the low level planners.
        self.low_level_planner_l = low_level_planner_l
        # Whether to use experience in the low level planner.
        self.is_xcbs = is_xcbs
        self.experience_reuse_strategy = CBSExperienceReuseStrategy.XCBS
        # Which conflicts to find and what constraints to create from them.
        self.conflict_type_to_constraint_types = conflict_type_to_constraint_types
        # Whether to impose soft constraints from other agents' paths.
        self.is_ecbs = is_ecbs
        self.num_agents = len(start_l)
        self.agent_color_l = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))
        self.start_state_pos_l = start_l
        self.goal_state_pos_l = goal_l
        if start_time_l is None:
            self.start_time_l = [0] * self.num_agents
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
            print(RED + 'Start or goal states are invalid.')
            print(self.start_state_pos_l)
            print(self.goal_state_pos_l, RESET)
            raise ValueError('Start or goal states are invalid.')
        # Open list.
        self.open_l = []

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
        # Return empty list if there are no paths.
        if len(paths_pos_l) == 0:
            return []
        max_t = max([len(path) for path in paths_pos_l])  # This should be the same for all agents.
        # Find conflicts.
        conflicts = []
        densification_factor = 2 if EdgeConflict in self.conflict_type_to_constraint_types.keys() else 1
        paths_pos_l_dense = densify_trajs(paths_pos_l, densification_factor)  # n elements of shape (H * densification_factor, q_dim)

        # Check collisions for all time steps at once.
        paths_pos_b_dense = torch.stack(paths_pos_l_dense)  # n_robots, H, q_dim
        # Change to (H, n_robots, q_dim) for the robot method.
        paths_pos_b_dense = paths_pos_b_dense.permute(1, 0, 2)
        collisions_pairwise_b, collision_points_b = self.reference_robot.check_rr_collisions(paths_pos_b_dense)  # Shapes (H, n_robots, n_robots) bool, (H, n_robots, n_robots, ws_dim) float
        # Check all time steps that have collisions in then.
        collision_indices = torch.nonzero(
            collisions_pairwise_b.int())  # Shape: (num_collisions, 3), each row [t_dense, agent_id_a, agent_id_b]
        for collision_index in collision_indices:
            t_dense, agent_id_a, agent_id_b = collision_index
            t_global_from = floor(t_dense / densification_factor)
            t_global_to = ceil(t_dense / densification_factor)
            t_dense, t_global_from, t_global_to = int(t_dense), int(t_global_from), int(t_global_to)
            # We set the conflict to happen at the beginning of the interpolated time interval,
            # but use the midpoint between interpolated states.
            midpoint_state_pos = collision_points_b[t_dense, agent_id_a, agent_id_b, :]
            agent_id_a, agent_id_b = int(agent_id_a.item()), int(agent_id_b.item())
            # Check which conflict types are requested.
            # Create a vertex conflict if the collision time is integral.
            if VertexConflict in self.conflict_type_to_constraint_types and t_global_from == t_global_to:
                conflicts.append(
                    VertexConflict([agent_id_a, agent_id_b],
                                   [
                                       paths_pos_l[agent_id_a][t_global_from],  # Corresponds to a graph vertex.
                                       paths_pos_l[agent_id_b][t_global_from]   # Corresponds to a graph vertex.
                                   ],
                                   int(t_global_from)))

            # Create an edge conflict if the collision time is not integral.
            if EdgeConflict in self.conflict_type_to_constraint_types and t_global_from != t_global_to:
                conflicts.append(
                    EdgeConflict([agent_id_a, agent_id_b],
                                 q_from_l=[
                                     paths_pos_l[agent_id_a][t_global_from],
                                     paths_pos_l[agent_id_b][t_global_from]
                                 ],
                                 q_to_l=[
                                     paths_pos_l[agent_id_a][t_global_to],
                                     paths_pos_l[agent_id_b][t_global_to]
                                 ],
                                 t_from=t_global_from,
                                 t_to=t_global_to))

            # Create a point conflict.
            # NOTE(yorai): this is normally called without a densification factor (=1), so the time from/to is the same.
            if PointConflict in self.conflict_type_to_constraint_types:
                conflicts.append(
                    PointConflict([agent_id_a, agent_id_b],
                                  p_l=[
                                        paths_pos_l_dense[agent_id_a][t_dense],
                                        paths_pos_l_dense[agent_id_b][t_dense]
                                  ],
                                  q_l=[
                                       midpoint_state_pos,
                                       midpoint_state_pos
                                  ],
                                  t_from=int(t_global_from),
                                  t_to=int(t_global_to)))
        return conflicts

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
                    start_state=self.start_state_pos_l[agent_id],  # None,  #
                    goal_state= self.goal_state_pos_l[agent_id],  # None,  #
                    colors=[self.agent_color_l[agent_id]],
                    constraints_l=constraints_l,
                    show_robot_in_image=show_robot_in_image
                )
            if output_fpath is None:
                output_fpath = os.path.join(self.results_dir, 'robot-traj.png')
                output_fpath = os.path.abspath(output_fpath)
            if output_fpath[-4:] != '.png':
                output_fpath += '.png'
            print(f'Saving image to: file://{output_fpath}')
            plt.axis('off')
            plt.savefig(output_fpath, dpi=100, bbox_inches='tight', pad_inches=0)
            return

        base_file_name = Path(os.path.basename(__file__)).stem
        if output_fpath is None:
            output_fpath = os.path.join(self.results_dir, f'{base_file_name}-robot-traj.gif')
            output_fpath = os.path.abspath(output_fpath)
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
        Plan a path from start to goal with constraints.
        """
        startt = time.time()
        # This success status will be updated by the search.
        success_status = TrialSuccessStatus.UNKNOWN
        # ======================
        # Create the root node.
        # ======================
        # Plan individual paths without constraints.
        root_creation_start_time = time.time()
        # Empty root node.
        root = SearchState([], [])
        for i in range(len(self.low_level_planner_l)):
            # If ECBS, then pass the paths of other agents to this one in the form of constraints.
            soft_constraint_l = []
            if self.is_ecbs:
                soft_constraint_l = self.create_soft_constraints_from_other_agents_paths(root, agent_id=i)

            planner_output = self.low_level_planner_l[i](self.start_state_pos_l[i],
                                                         self.goal_state_pos_l[i],
                                                         constraints_l=soft_constraint_l)
            # Check for planning failure in root creation.
            if planner_output.trajs_final_free_idxs.shape[0] == 0:
                print("Failed to find valid paths in root CT node.")
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                state = root
                break

            # Update the root node.
            ix_best_traj = planner_output.idx_best_traj
            root.path_bl.append(planner_output.trajs_final)
            root.ix_best_path_in_batch_l.append(ix_best_traj)

            # Check for runtime limit reached in root creation.
            if time.time() - startt > runtime_limit:
                print('Runtime limit reached in root creation.')
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                state = root
                break

        if success_status == TrialSuccessStatus.UNKNOWN:
            root.update_g_l2()
            conflict_l = self.get_conflicts(root)
            root.conflict_l = conflict_l
            # Create the open list.
            self.open_l.append(root)
            print(f'Root creation time: {time.time() - root_creation_start_time:.2f}s')

        # ======================
        # Start the search.
        # ======================
        num_ct_expansions = 0
        while success_status == TrialSuccessStatus.UNKNOWN:
            # If the open list is empty, return None.
            if not self.open_l:
                print('Open list is empty. NO SOLUTION.')
                success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
                break

            # Sort the CT.
            # Optionally replace the line below with `self.open_l.sort(key=lambda x: x.g)` to optimize for cost.
            self.open_l.sort(key=lambda x: len(x.conflict_l))
            # Get the first node from the open list.
            state = self.open_l.pop(0)
            # Check if the conflict set is empty.
            if not state.conflict_l:
                print('No conflicts found in CT node. GOAL.')
                success_status = TrialSuccessStatus.SUCCESS
                break

            # Expand the search tree.
            self.expand(state)
            num_ct_expansions += 1
            if time.time() - startt > runtime_limit:
                print('Runtime limit reached.')
                success_status = TrialSuccessStatus.FAIL_RUNTIME_LIMIT
                break

        # Return the best paths. Smoothed and padded to the same length accounting for start times.
        best_path_l = [state.path_bl[i][ix_best_path_in_batch].squeeze(0) for i, ix_best_path_in_batch in
                       enumerate(state.ix_best_path_in_batch_l)]

        best_path_l = global_pad_paths(best_path_l, self.start_time_l)
        # Return the best paths, the number of CT expansions, success status, and number of collisions in the solution.
        return best_path_l, num_ct_expansions, success_status, len(state.conflict_l)

    def expand(self, state: SearchState):
        """
        Expand the search tree.
        """
        # Choose a conflict to turn into constraints.
        conflict = state.conflict_l[0]
        constraints = convert_conflicts_to_constraints(conflict, self.conflict_type_to_constraint_types)
        # Create a CT node for each constraint tuple (agent_id, constraint).
        for agent_id, constraint in constraints:
            # Update the constraint time to account for agents starting after t=0.
            constraint.t_range_l = [(t_range[0] - self.start_time_l[agent_id],
                                     t_range[1] - self.start_time_l[agent_id])
                                    for t_range in constraint.t_range_l]
            # Clamp down to the maximum time of the paths.
            constraint.t_range_l = [(max(0, min(t_range[0], len(state.path_bl[agent_id][0]) - 1)),
                                     min(len(state.path_bl[agent_id][0]) - 1, t_range[1]))
                                    for t_range in constraint.t_range_l]
            # Create new state.
            new_state = state.get_copy()
            # Add the constraint.
            new_state.add_constraint(agent_id, constraint)  # In local time to agent.
            # Create a PlanningContext object for the agent which includes the current trajectories of all other agents.
            # Plan the path with the constraint. Add the previous path as experience as a seed if allowed.
            agent_constraint_l = new_state.constraints[agent_id].copy()

            # Set soft constraints from paths of other agents, if allowed.
            if self.is_ecbs:
                soft_constraint_l = self.create_soft_constraints_from_other_agents_paths(new_state, agent_id)
                agent_constraint_l.extend(soft_constraint_l)

            # Set experience, if allowed.
            agent_experience = None
            if self.is_xcbs:
                if self.experience_reuse_strategy == CBSExperienceReuseStrategy.XCBS:
                    agent_experience = PathBatchExperience(new_state.path_bl[agent_id])
                else:
                    raise ValueError(f'Invalid experience reuse strategy {self.experience_reuse_strategy}.')
            planner_output = self.low_level_planner_l[agent_id](self.start_state_pos_l[agent_id],
                                                                self.goal_state_pos_l[agent_id],
                                                                constraints_l=agent_constraint_l,
                                                                experience=agent_experience)

            # Check if the planner found a valid path.
            if len(planner_output.trajs_final_free_idxs) == 0:
                print(RED + 'Failed to find valid path in CT node.' + RESET)
                return  # Skip this node.

            new_state.path_bl[agent_id] = planner_output.trajs_final

            if self.low_level_choose_path_from_batch_strategy == 'least_cost':
                ix_best_traj = planner_output.idx_best_traj
                new_state.ix_best_path_in_batch_l[agent_id] = ix_best_traj
                # Find and set conflicts.
                conflict_l = self.get_conflicts(new_state)
                new_state.conflict_l = conflict_l

            elif self.low_level_choose_path_from_batch_strategy == 'least_collisions':
                new_state.conflict_l = None
                for ix_traj in planner_output.trajs_final_free_idxs:
                    temp_state = new_state.get_copy()
                    temp_state.ix_best_path_in_batch_l[agent_id] = ix_traj
                    conflict_l = self.get_conflicts(temp_state)
                    if new_state.conflict_l is None:
                        new_state.ix_best_path_in_batch_l[agent_id] = ix_traj
                        new_state.conflict_l = conflict_l
                    if len(conflict_l) < len(new_state.conflict_l):
                        new_state.ix_best_path_in_batch_l[agent_id] = ix_traj
                        new_state.conflict_l = conflict_l
                        print(f'Found a path with fewer conflicts: {len(conflict_l)}')
            else:
                raise ValueError('Invalid low level choose-path-from-batch strategy.')

            # Update the cost to reach this node.
            new_state.update_g_l2()
            print("New state cost num conflicts:", new_state.g, len(new_state.conflict_l), "\n\n\n\n")
            # Add the new state to the open list.
            self.open_l.append(new_state)

    def create_soft_constraints_from_other_agents_paths(self,
                                                        state: SearchState,
                                                        agent_id: int) -> List[MultiPointConstraint]:
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
                    # The last timestep index for this agent is the length of its path - 1.
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
