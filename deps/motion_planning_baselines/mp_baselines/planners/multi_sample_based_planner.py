from copy import copy

from mp_baselines.planners.multi_processing import MultiProcessor


class MultiSampleBasedPlanner:

    def __init__(
            self,
            planner,
            n_trajectories=2,
            optimize_sequentially=False,
            **kwargs
    ):
        self.planner = planner
        self.n_trajectories = n_trajectories
        self.multi_processer = None
        if not optimize_sequentially:
            self.planners_l = [copy(planner) for _ in range(n_trajectories)]
            self.multi_processer = MultiProcessor(**kwargs)

    def optimize(self, **kwargs):
        if self.multi_processer is not None:
            # optimize in parallel
            for p in self.planners_l:  # queue up multiple tasks
                self.multi_processer.run(p.optimize, **kwargs)
            trajs_l = self.multi_processer.wait()  # get all results
        else:
            # optimize sequentially
            trajs_l = []
            for _ in range(self.n_trajectories):  # queue up multiple tasks
                traj = self.planner.optimize(**kwargs)
                trajs_l.append(traj)
        return trajs_l

    @property
    def start_state_pos(self):
        return self.planner.start_state_pos

    @property
    def goal_state_pos(self):
        return self.planner.goal_state_pos
