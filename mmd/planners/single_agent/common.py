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


class PlannerOutput:
    def __init__(self):
        self.trajs_iters = None
        self.trajs_final = None
        self.trajs_final_coll = None
        self.trajs_final_coll_idxs = None
        self.trajs_final_free = None
        self.trajs_final_free_idxs = None
        self.success_free_trajs = None
        self.fraction_free_trajs = None
        self.collision_intensity_trajs = None
        self.idx_best_traj = None  # In trajs_final.
        self.traj_final_free_best = None
        self.cost_best_free_traj = None
        self.cost_smoothness = None
        self.cost_path_length = None
        self.cost_all = None
        self.variance_waypoint_trajs_final_free = None
        self.t_total = None
        # The constraints set used for this call.
        self.constraints_l = None
