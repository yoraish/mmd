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
from abc import ABC
import torch


class Experience(ABC):
    def __init__(self):
        pass


class PathExperience(Experience):
    def __init__(self, path: torch.Tensor):
        super().__init__()
        # This is the experience path. This object is a torch tensor commonly of shape (B, H, q_dim).
        # B is the batch size, H is the horizon, and q_dim is the dimension of the configuration
        # (including perhaps dynamics).
        # A common case of disk robots this is (50, 64, 4). The final dimension is x, y, xdot, ydot.
        self.path = path


class PathBatchExperience(Experience):
    def __init__(self, path_b: torch.Tensor):
        super().__init__()
        # This is the experience path batch. This object is a torch tensor commonly of shape (B, H, q_dim).
        # B is the batch size, H is the horizon, and q_dim is the dimension of the configuration
        # (including perhaps dynamics).
        # In the common case of disk robots this is (50, 64, 4). x, y, xdot, ydot.
        self.path_b = path_b
        