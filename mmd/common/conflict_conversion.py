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

from typing import List, Tuple, Dict, Type
# MMD imports.
from mmd.common.conflicts import Conflict, VertexConflict, EdgeConflict, PointConflict
from mmd.common.constraints import MultiPointConstraint, EdgeConstraint, VertexConstraint, Constraint
from mmd.config.mmd_params import MMDParams as params


def convert_conflicts_to_constraints(conflict: Conflict,
                                     conflict_type_to_constraint_types: Dict[Type[Conflict], Type[Constraint]],
                                     t_pad: int = 2,
                                     ) -> List[Tuple[int, MultiPointConstraint]]:
    """
    Convert a conflict to a list of constraints.
    """
    # Get the type of the constraint to create for this type of conflict.
    constraints = []
    if isinstance(conflict, PointConflict):
        # def conflicts_to_constraints(self, conflicts: List[VertexConflict]) -> List[Tuple[int, Constraint]]:
        """
        Convert a list of conflicts to a list of constraints.
        """
        if MultiPointConstraint in conflict_type_to_constraint_types[PointConflict]:
            for agent_id in conflict.agent_ids:
                constraints.append(
                    (agent_id, MultiPointConstraint(
                        q_l=[conflict.agent_id_to_q[agent_id]],
                        t_range_l=[(conflict.t_from - t_pad,
                                    conflict.t_to + t_pad)],
                        radius_l=[params.vertex_constraint_radius])
                     )
                )

        else:
            raise NotImplementedError()

    elif isinstance(conflict, EdgeConflict):
        if EdgeConstraint in conflict_type_to_constraint_types[EdgeConflict]:
            for agent_id in conflict.agent_ids:
                constraints.append(
                    (agent_id, EdgeConstraint(
                        q_from=conflict.agent_id_to_q_from[agent_id],
                        q_to=conflict.agent_id_to_q_to[agent_id],
                        t_from=conflict.t_from,
                        t_to=conflict.t_to)))
        else:
            raise NotImplementedError()

    elif isinstance(conflict, VertexConflict):
        if VertexConstraint in conflict_type_to_constraint_types[VertexConflict]:
            for agent_id in conflict.agent_ids:
                constraints.append(
                    (agent_id, VertexConstraint(
                        q=conflict.q_map[agent_id],
                        t=conflict.t)))
        else:
            raise NotImplementedError()

    return constraints
