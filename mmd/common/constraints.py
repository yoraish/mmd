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

# General includes.
from abc import ABC, abstractmethod
from typing import Tuple, List
import torch

# Project includes.
from mmd.config import MMDParams as params


class Constraint(ABC):
    """
    A class holding within it information for defining a constraint.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_t_range_l(self):
        pass


class MultiPointConstraint(Constraint):
    """
    A class holding within it information for defining a batch of constraints. Each constraint has an associated time
    range in which it is active, a configuration for its center, and a radius in configuration space.
    """
    def __init__(self, q_l: List[torch.Tensor],
                 t_range_l: List[Tuple[int, int]],
                 radius_l: List[float] = None,
                 is_soft: bool = False):
        super().__init__()
        # The list of configurations that are constrained.
        self.q_l = q_l
        # The time range in which the constraint is active. Inclusive on boundary.
        self.t_range_l = t_range_l
        # The radius of the constraint, in configuration space.
        self.radius_l = [params.vertex_constraint_radius] * len(q_l) if radius_l is None else radius_l
        # The weight of the constraint. This may be used as the guide gradient scaling factor.
        self.is_soft = is_soft

    def get_q_l(self) -> List[torch.Tensor]:
        return self.q_l

    def get_t_range_l(self):
        return self.t_range_l

    def get_radius_l(self):
        return self.radius_l

    def get_is_soft(self):
        return self.is_soft

    def get_copy(self):
        constraint_new = MultiPointConstraint(self.q_l.copy(), self.t_range_l.copy(), self.radius_l.copy(), self.is_soft)
        return constraint_new

    def __str__(self):
        return f"q_l: {self.q_l}, t_range_l: {self.t_range_l}, radius_l: {self.radius_l}, is_soft: {self.is_soft}"

    def __repr__(self):
        return f"q_l: {self.q_l}, t_range_l: {self.t_range_l}, radius_l: {self.radius_l}, is_soft: {self.is_soft}"


class VertexConstraint(Constraint):
    """
    A class holding within it information for defining a vertex constraint.
    """
    def __init__(self, q: torch.Tensor,
                 t: int) -> None:
        super().__init__()
        # The configuration that is constrained.
        self.q = q
        # The time at which the constraint is active.
        self.t_range_l = [[t, t]]

    def get_q(self) -> torch.Tensor:
        return self.q

    def get_t_range_l(self):
        return self.t_range_l

    def get_copy(self):
        constraint_new = VertexConstraint(self.q.clone(), self.t_range_l[0][0])
        return constraint_new

    def __repr__(self):
        return f"VertexConstraint at q: {self.q}, t_range_l: {self.t_range_l}"


class EdgeConstraint(Constraint):
    """
    A class holding within it information for defining an edge constraint.
    """
    def __init__(self, q_from: torch.Tensor, t_from: int,
                 q_to: torch.Tensor, t_to: int) -> None:
        super().__init__()
        # The configurations that are constrained.
        self.q_from = q_from
        self.q_to = q_to
        # The time at which the constraint is active.
        self.t_range_l = [[t_from, t_to]]

    def get_q_from(self) -> torch.Tensor:
        return self.q_from

    def get_q_to(self) -> torch.Tensor:
        return self.q_to

    def get_t_range_l(self):
        return self.t_range_l

    def get_copy(self):
        constraint_new = EdgeConstraint(self.q_from.clone(),
                                        self.t_range_l[0][0],
                                        self.q_to.clone(),
                                        self.t_range_l[0][1])
        return constraint_new

    def __repr__(self):
        return f"EdgeConstraint from q: {self.q_from}, to q: {self.q_to}, t_range_l: {self.t_range_l}"
