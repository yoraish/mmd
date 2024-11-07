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
# General imports.
import torch

# Project imports.
from mp_baselines.planners.base import MPPlanner


class IdentityPlanner(MPPlanner):

    def __init__(
            self,
            fixed_path: torch.Tensor = None,  # Shape (num_points, 2).
            tensor_args: dict = None,
            **kwargs
    ):
        assert fixed_path is not None

        super().__init__(name="IdentityPlanner", tensor_args=tensor_args)

        self.start_state_pos = fixed_path[0]
        self.goal_state_pos = fixed_path[-1]
        self.fixed_path = fixed_path

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """
        return self.fixed_path

    def render(self, ax, **kwargs):
        ax.plot(self.fixed_path[:, 0], self.fixed_path[:, 1], color='r', linewidth=2, marker='o')
