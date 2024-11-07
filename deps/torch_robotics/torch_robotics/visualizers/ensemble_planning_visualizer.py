import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from torch_robotics.torch_utils.torch_utils import to_numpy
from torch_robotics.tasks.tasks_ensemble import PlanningTaskEnsemble
from .planning_visualizer import create_fig_and_axes
import matplotlib.collections as mcoll


class PlanningVisualizerEnsemble:

    def __init__(self,
                 task_ensmeble: PlanningTaskEnsemble,
                 planner=None):
        self.task_ensmeble = task_ensmeble
        self.env = self.task_ensmeble.env
        self.robot = self.task_ensmeble.tasks[0].robot
        self.planner = planner

        self.colors = {'collision': 'black', 'free': 'orange'}
        self.colors_robot = {'collision': 'black', 'free': 'darkorange'}
        self.cmaps = {'collision': 'Greys', 'free': 'Oranges'}
        self.cmaps_robot = {'collision': 'Greys', 'free': 'YlOrRd'}

    def render_robot_trajectories(self, fig=None, ax=None, render_planner=False, trajs=None, traj_best=None, **kwargs):
        if fig is None or ax is None:
            fig, ax = create_fig_and_axes(dim=self.env.dim)

        if render_planner:
            self.planner.render(ax)
        self.env.render(ax)
        if trajs is not None:
            _, trajs_coll_idxs, _, trajs_free_idxs, _ = self.task.get_trajs_collision_and_free(trajs,
                                                                                               return_indices=True)
            kwargs['colors'] = []
            for i in range(len(trajs_coll_idxs) + len(trajs_free_idxs)):
                kwargs['colors'].append(self.colors['collision'] if i in trajs_coll_idxs else self.colors['free'])
        self.robot.render_trajectories(ax, trajs=trajs, **kwargs)
        if traj_best is not None:
            kwargs['colors'] = ['blue']
            self.robot.render_trajectories(ax, trajs=traj_best.unsqueeze(0), **kwargs)

        return fig, ax

    def animate_robot_trajectories(
            self, trajs=None, start_state=None, goal_state=None,
            plot_trajs=False,
            n_frames=10,
            constraints=None,
            **kwargs
    ):
        if trajs is None:
            return

        assert trajs.ndim == 3
        B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_selection = trajs[:, idxs, :]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            """
            Draw all the robots at step i for all trajectories.
            Also add a robot configuration at any constraints.
            """
            print(idxs[i], "/", H, end="\r")
            ax.clear()
            ax.set_title(f"step: {idxs[i]}/{H - 1}")
            if plot_trajs:
                self.render_robot_trajectories(
                    fig=fig, ax=ax, trajs=trajs, start_state=start_state, goal_state=goal_state, **kwargs
                )
            else:
                self.env.render(ax)

            # TODO - implement batched version
            qs = trajs_selection[:, i, :]  # batch, q_dim
            if qs.ndim == 1:
                qs = qs.unsqueeze(0)  # interface (batch, q_dim)
            print(qs.shape)
            for i, q in enumerate(qs):
                self.robot.render(
                    ax, q=q,
                    color=self.colors_robot['collision'] if self.task.compute_collision(q, margin=0.0) else
                    self.colors_robot['free'],
                    arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                    cmap=self.cmaps['collision'] if self.task.compute_collision(q, margin=0.0) else self.cmaps['free'],
                    **kwargs
                )

            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')
            # Also add any constraints to the plot, if those exist and are active at this step i.
            step = idxs[i]
            if constraints is not None:
                for constraint in constraints:
                    # For Constraint object passed.
                    if constraint.get_t_range()[0] <= step <= constraint.get_t_range()[1]:
                        self.robot.render(ax, constraint.get_q(), color='red', cmap='Reds')
                    # For list of CostCosntaint objects passed.
                    # if constraint.traj_range[0] <= step <= constraint.traj_range[1]:
                    #     self.robot.render(ax, constraint.q, color='red', cmap='Reds')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_multi_robot_trajectories(
            self, trajs_l=None, start_state_l=None, goal_state_l=None,
            plot_trajs=False,
            n_frames=10,
            constraints=None,
            colors=None,
            **kwargs
    ):
        assert len(colors) == len(trajs_l)

        assert trajs_l[0].ndim == 3
        B, H, D = trajs_l[0].shape

        idxs = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_l_selection = []
        for trajs in trajs_l:
            trajs_selection = trajs[:, idxs, :]
            trajs_l_selection.append(trajs_selection)

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            """
            Draw all the robots at step i for all trajectories.
            Also add a robot configuration at any constraints.
            """
            print(idxs[i], "/", H, end="\r")
            ax.clear()
            ax.set_title(f"step: {idxs[i]}/{H - 1}")
            if plot_trajs:
                for trajs, color, start_state, goal_state in zip(trajs_l_selection, colors, start_state_l,
                                                                 goal_state_l):
                    self.render_robot_trajectories(
                        fig=fig,
                        ax=ax,
                        trajs=trajs,
                        start_state=start_state,
                        goal_state=goal_state,
                        colors=[color] * len(trajs),
                        **kwargs
                    )
            else:
                self.env.render(ax)

            for trajs_selection, color, start_state, goal_state in zip(trajs_l_selection, colors, start_state_l,
                                                                       goal_state_l):
                qs = trajs_selection[:, i, :]  # batch, q_dim
                if qs.ndim == 1:
                    qs = qs.unsqueeze(0)  # interface (batch, q_dim)
                for q in qs:
                    q_tail = []
                    for di in range(5):
                        q_tail.append(trajs_selection[:, i - di, :].squeeze())
                    kwargs['q_tail'] = q_tail
                    self.robot.render(
                        ax, q=q,
                        color=color,
                        arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.,
                        cmap=self.cmaps['collision'] if self.task.compute_collision(q, margin=0.0) else self.cmaps[
                            'free'],
                        **kwargs
                    )

                if start_state is not None:
                    self.robot.render(ax, start_state, color='green', cmap='Greens')
                if goal_state is not None:
                    self.robot.render(ax, goal_state, color='purple', cmap='Purples')
                # Also add any constraints to the plot, if those exist and are active at this step i.
                step = idxs[i]
                if constraints is not None:
                    for constraint in constraints:
                        # For Constraint object passed.
                        if constraint.get_t_range()[0] <= step <= constraint.get_t_range()[1]:
                            self.robot.render(ax, constraint.get_q(), color='red', cmap='Reds')
                        # For list of CostCosntaint objects passed.
                        # if constraint.traj_range[0] <= step <= constraint.traj_range[1]:
                        #     self.robot.render(ax, constraint.q, color='red', cmap='Reds')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def animate_opt_iters_robots(
            self, trajs=None, traj_best=None, start_state=None, goal_state=None,
            n_frames=10,
            **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, ax = create_fig_and_axes(dim=self.env.dim)

        def animate_fn(i):
            ax.clear()
            ax.set_title(f"iter: {idxs[i]}/{S - 1}")
            self.render_robot_trajectories(
                fig=fig, ax=ax, trajs=trajs_selection[i],
                traj_best=traj_best if i == n_frames - 1 else None,
                start_state=start_state, goal_state=goal_state, **kwargs
            )
            if start_state is not None:
                self.robot.render(ax, start_state, color='green', cmap='Greens')
            if goal_state is not None:
                self.robot.render(ax, goal_state, color='purple', cmap='Purples')

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)

    def plot_joint_space_state_trajectories(
            self,
            fig=None, axs=None,
            trajs=None,
            traj_best=None,
            pos_start_state=None, pos_goal_state=None,
            vel_start_state=None, vel_goal_state=None,
            set_joint_limits=True,
            **kwargs
    ):
        if trajs is None:
            return
        trajs_np = to_numpy(trajs)

        assert trajs_np.ndim == 3
        B, H, D = trajs_np.shape

        # Separate trajectories in collision and free (not in collision)
        trajs_coll, trajs_free = self.task.get_trajs_collision_and_free(trajs)

        trajs_coll_pos_np = to_numpy([])
        trajs_coll_vel_np = to_numpy([])
        if trajs_coll is not None:
            trajs_coll_pos_np = to_numpy(self.robot.get_position(trajs_coll))
            trajs_coll_vel_np = to_numpy(self.robot.get_velocity(trajs_coll))

        trajs_free_pos_np = to_numpy([])
        trajs_free_vel_np = to_numpy([])
        if trajs_free is not None:
            trajs_free_pos_np = to_numpy(self.robot.get_position(trajs_free))
            trajs_free_vel_np = to_numpy(self.robot.get_velocity(trajs_free))

        if pos_start_state is not None:
            pos_start_state = to_numpy(pos_start_state)
        if vel_start_state is not None:
            vel_start_state = to_numpy(vel_start_state)
        if pos_goal_state is not None:
            pos_goal_state = to_numpy(pos_goal_state)
        if vel_goal_state is not None:
            vel_goal_state = to_numpy(vel_goal_state)

        if fig is None or axs is None:
            fig, axs = plt.subplots(self.robot.q_dim, 2, squeeze=False)
        axs[0, 0].set_title('Position')
        axs[0, 1].set_title('Velocity')
        axs[-1, 0].set_xlabel('Timesteps')
        axs[-1, 1].set_xlabel('Timesteps')
        timesteps = np.arange(H).reshape(1, -1)
        for i, ax in enumerate(axs):
            for trajs_filtered, color in zip(
                    [(trajs_coll_pos_np, trajs_coll_vel_np), (trajs_free_pos_np, trajs_free_vel_np)],
                    ['black', 'orange']):
                # Positions and velocities
                for j, trajs_filtered_ in enumerate(trajs_filtered):
                    if trajs_filtered_.size > 0:
                        timesteps_ = np.repeat(timesteps, trajs_filtered_.shape[0], axis=0)
                        plot_multiline(ax[j], timesteps_, trajs_filtered_[..., i], color=color, **kwargs)

            if traj_best is not None:
                traj_best_pos = self.robot.get_position(traj_best)
                traj_best_vel = self.robot.get_velocity(traj_best)
                traj_best_pos_np = to_numpy(traj_best_pos)
                traj_best_vel_np = to_numpy(traj_best_vel)
                plot_multiline(ax[0], timesteps, traj_best_pos_np[..., i].reshape(1, -1), color='blue', **kwargs)
                plot_multiline(ax[1], timesteps, traj_best_vel_np[..., i].reshape(1, -1), color='blue', **kwargs)

            # # Start and goal
            # if pos_start_state is not None:
            #     ax[0].scatter(0, pos_start_state[i], color='green')
            # if vel_start_state is not None:
            #     ax[1].scatter(0, vel_start_state[i], color='green')
            # if pos_goal_state is not None:
            #     ax[0].scatter(H - 1, pos_goal_state[i], color='purple')
            # if vel_goal_state is not None:
            #     ax[1].scatter(H - 1, vel_goal_state[i], color='purple')
            # Y label
            ax[0].set_ylabel(f'q_{i}')
            # Set limits
            if set_joint_limits:
                ax[0].set_ylim(self.robot.q_min_np[i], self.robot.q_max_np[i])
                # ax[1].set_ylim(self.robot.q_vel_min_np[i], self.robot.q_vel_max_np[i])

        return fig, axs

    def animate_opt_iters_joint_space_state(
            self, trajs=None, traj_best=None, n_frames=10, **kwargs
    ):
        # trajs: steps, batch, horizon, q_dim
        if trajs is None:
            return

        assert trajs.ndim == 4
        S, B, H, D = trajs.shape

        idxs = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_selection = trajs[idxs]

        fig, axs = self.plot_joint_space_state_trajectories(trajs=trajs_selection[0], **kwargs)

        def animate_fn(i):
            [ax.clear() for ax in axs.ravel()]
            fig.suptitle(f"iter: {idxs[i]}/{S - 1}")
            self.plot_joint_space_state_trajectories(
                fig=fig, axs=axs,
                trajs=trajs_selection[i], **kwargs
            )
            if i == n_frames - 1 and traj_best is not None:
                self.plot_joint_space_state_trajectories(
                    fig=fig, axs=axs,
                    trajs=trajs_selection[i],
                    traj_best=traj_best, **kwargs
                )

        create_animation_video(fig, animate_fn, n_frames=n_frames, **kwargs)


def create_animation_video(fig, animate_fn, anim_time=5, n_frames=100, video_filepath='video.gif', **kwargs):
    str_start = "Creating animation"
    # if ".gif" in video_filepath:
    #     video_filepath = video_filepath.replace(".gif", ".gif")
    print(f'{str_start}...')
    ani = FuncAnimation(
        fig,
        animate_fn,
        frames=n_frames,
        interval=anim_time * 1000 / n_frames,
        repeat=False
    )
    print(f'...finished {str_start}')

    str_start = "Saving video..."
    print(f'{str_start}...')
    ani.save(os.path.join(video_filepath), fps=max(1, int(n_frames / anim_time)), dpi=90)
    print(f'...finished {str_start}')


def plot_multiline(ax, X, Y, color='blue', linestyle='solid', **kwargs):
    segments = np.stack((X, Y), axis=-1)
    line_segments = mcoll.LineCollection(segments, colors=[color] * len(segments), linestyle=linestyle)
    ax.add_collection(line_segments)
    points = np.reshape(segments, (-1, 2))
    ax.scatter(points[:, 0], points[:, 1], color=color, s=2 ** 2)
