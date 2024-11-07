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
from typing import List, Tuple
from abc import ABC, abstractmethod


class Conflict(ABC):
    """
    A class holding within it information for defining a conflict.
    """
    def __init__(self):
        self.time_interval = None  # Should be two-element list.

    @abstractmethod
    def get_t_range(self) -> Tuple[int, int]:
        pass


class VertexConflict(Conflict):
    """
    A class holding within it information for defining a conflict.
    """
    def __init__(self, agent_ids: List, q_l: List, t: int):
        super().__init__()
        self.agent_ids = agent_ids
        self.q_map = {agent_id: q for agent_id, q in zip(agent_ids, q_l)}
        self.t = t

    def get_t_range(self) -> Tuple[int, int]:
        return self.t, self.t

    def __repr__(self):
        return (f"VertexConflict(agent_ids={self.agent_ids}, \n" +
                f"q_l={self.q_map}, \n" +
                f"t={self.t})")


class EdgeConflict(Conflict):
    """
    A class holding within it information for defining a conflict.
    """
    def __init__(self, agent_ids: List,
                 q_from_l: List, q_to_l: List,
                 t_from: int, t_to: int,
                 ):
        super().__init__()
        self.agent_ids = agent_ids
        self.agent_id_to_q_from = {agent_id: q_from for agent_id, q_from in zip(agent_ids, q_from_l)}
        self.agent_id_to_q_to = {agent_id: q_to for agent_id, q_to in zip(agent_ids, q_to_l)}
        self.t_from = t_from
        self.t_to = t_to

    def get_t_range(self) -> Tuple[int, int]:
        return self.t_from, self.t_to

    def __repr__(self):
        return (f"EdgeConflict(agent_ids={self.agent_ids}, \n" +
                f"    q_from_l={self.agent_id_to_q_from}, \n" +
                f"    q_to_l={self.agent_id_to_q_to}, \n" +
                f"    t_from={self.t_from}, \n" +
                f"    t_to={self.t_to})")


class PointConflict(Conflict):
    """
    A class holding within it information for defining a conflict.
    """
    def __init__(self, agent_ids: List, q_l: List, p_l: List, t_from: int, t_to: int):
        super().__init__()
        self.agent_ids = agent_ids
        self.agent_id_to_p = {agent_id: p for agent_id, p in zip(agent_ids, p_l)}
        self.agent_id_to_q = {agent_id: q for agent_id, q in zip(agent_ids, q_l)}
        self.t_from = t_from
        self.t_to = t_to

    def get_t_range(self) -> Tuple[int, int]:
        return self.t_from, self.t_to

    def __repr__(self):
        return (f"PointConflict(agent_ids={self.agent_ids}, \n"
                f"    q_l={self.agent_id_to_q}, \n" +
                f"    p_l={self.agent_id_to_p}, \n" +
                f"    t_from={self.t_from}, \n" +
                f"    t_to={self.t_to})")

