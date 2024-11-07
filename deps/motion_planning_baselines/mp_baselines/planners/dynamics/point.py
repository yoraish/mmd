import numpy as np
import torch


class PointParticleDynamics:
    def __init__(
            self,
            rollout_steps=None,
            control_dim=2,
            state_dim=2,
            dt=0.01,
            discount=1.0,
            deterministic=True,
            start_state=None,
            goal_state=None,
            ctrl_min: list = None,
            ctrl_max: list = None,
            control_type='velocity',
            dyn_std=np.zeros(4,),
            c_weights=None,
            verbose=False,
            tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = {'device': torch.device('cpu'), 'dtype': torch.float32}
        self.tensor_args = tensor_args

        self.control_dim = control_dim
        if control_type == 'velocity':
            self.state_dim = state_dim
        elif control_type == 'acceleration':
            self.state_dim = state_dim * 2
        else:
            raise IOError('control_type "{}" not recognized'.format(control_type))
        if c_weights is None:
            self._c_weights = {
                'pos': 10.,
                'vel': 10.,
                'ctrl': 0.,
                'pos_T': 10.,
                'vel_T': 0.,
            }
        else:
            self._c_weights = c_weights

        assert len(ctrl_min) == self.control_dim
        assert len(ctrl_max) == self.control_dim
        self.ctrl_min = torch.tensor(ctrl_min).to(**tensor_args)
        self.ctrl_max = torch.tensor(ctrl_max).to(**tensor_args)

        self.discount_seq = self._get_discount_seq(discount, rollout_steps)

        if start_state is not None:
            self.start_state = torch.from_numpy(start_state).to(**self.tensor_args)
        else:
            self.start_state = torch.zeros(self.state_dim, **self.tensor_args)
        self.state = self.start_state.clone()
        if goal_state is not None:
            if isinstance(goal_state, np.ndarray):
                self.goal_state = torch.from_numpy(goal_state).to(**self.tensor_args)
            elif isinstance(goal_state, torch.Tensor):
                self.goal_state = goal_state
            else:
                raise IOError
        else:
            self.goal_state = torch.zeros(self.state_dim, **self.tensor_args)
        self.rollout_steps = rollout_steps
        self.dt = dt
        self.dyn_std = torch.from_numpy(dyn_std).to(**self.tensor_args)
        self.control_type = control_type
        self.discount = discount
        self.verbose = verbose
        self.deterministic = deterministic

    @property
    def state(self):
        return self._state.clone().detach()

    @state.setter
    def state(self, value):
        self._state = value

    def reset(self):
        self.state = self.start_state.clone()
        cost, _ = self.traj_cost(
            self.state.reshape(1, 1, 1, -1),
            torch.zeros(1, 1, 1, self.control_dim),
        )
        return self.state, cost

    def step(self, action):
        state = self.state.reshape(1, 1, -1)
        action = action.reshape(1, 1, -1)
        self.state = self.dynamics(state, action)

        state = state.reshape(1, 1, 1, -1)
        action = action.reshape(1, 1, 1, -1)
        cost, _ = self.traj_cost(state, action)

        return self.state.squeeze(), cost.squeeze()

    def dynamics(
            self,
            x,
            u_s,
            use_crn=False,
    ):
        num_ctrl_samples, num_state_samples = x.size(0), x.size(1)
        ctrl_dim = u_s.size(-1)
        # clamp controls
        u = u_s.clamp(min=self.ctrl_min, max=self.ctrl_max)

        if self.deterministic:
            xdot = torch.cat(
                (x[..., self.state_dim:], u),
                dim=2,
            )
        else:
            # Noise in control channel
            if use_crn:
                noise = self.dyn_std * torch.randn(
                    num_state_samples,
                    ctrl_dim,
                    **self.tensor_args,
                )
            else:
                noise = self.dyn_std * torch.randn(
                    num_ctrl_samples,
                    num_state_samples,
                    ctrl_dim,
                    **self.tensor_args,
                )

            u_s = u + noise
            xdot = torch.cat(
                (x[..., self.state_dim:], u_s),
                dim=2,
            )
        x_next = x + xdot * self.dt
        return x_next

    def render(self, state=None, mode='human'):
        pass

    def _get_discount_seq(self, discount, rollout_steps):
        # Discount factors
        discount_seq = torch.cumprod(
            torch.ones(rollout_steps, **self.tensor_args) * discount,
            dim=0,
        )
        discount_seq /= discount  # start with weight one
        return discount_seq

    def traj_cost(
            self,
            X_in, U_in,
            **observation
    ):
        """
        Implements quadratic trajectory cost.
        Args
        ----
        X_in : Tensor
            State trajectories, of shape
                [steps+1, num_ctrl_samples, num_state_samples, state dim]
        U_in : Tensor
            Control trajectories, of shape
                [steps+1, num_ctrl_samples, num_state_samples, control dim]
        Returns
        -------
        cost : Tensor
            Trajectory costs, of shape [num_ctrl_samples, num_state_samples]
        """

        rollout_steps, num_ctrl_samples, num_state_samples, state_dim = X_in.shape

        batch_size = num_ctrl_samples * num_state_samples
        # New shape: X: [batch, steps+1, particles, state_dim]
        X = X_in.view(
            rollout_steps,
            batch_size,
            state_dim,
        ).transpose(0, 1)

        U = U_in.view(
            rollout_steps,
            batch_size,
            self.control_dim,
        ).transpose(0, 1)

        goal_state = observation.get('goal_state', self.goal_state)
        cost = observation.get('cost', None)

        if cost is not None:
            full_traj = torch.cat((X, U), dim=-1)
            energy_cost = cost.eval(full_traj).sum(-1)  # sum over the horizon
        else:
            energy_cost = 0.

        # Distance to goal
        dX = X - goal_state[..., :self.state_dim]
        dX_final = dX[:, -1, :]

        # Discounted Costs
        discount_seq = self.discount_seq.unsqueeze(0)
        pos_cost = (torch.square(dX[..., :self.state_dim]) * self._c_weights['pos']).sum(-1) * discount_seq
        vel_cost = (torch.square(dX[..., self.state_dim:self.control_dim]) * self._c_weights['vel']).sum(-1) * discount_seq
        control_cost = (torch.square(U) * self._c_weights['ctrl']).sum(-1) * discount_seq
        terminal_cost = (torch.square(dX_final) * self._c_weights['pos_T']).sum(-1) * discount_seq[..., -1]

        # Sum along trajectory timesteps
        pos_cost = pos_cost.sum(1)
        vel_cost = vel_cost.sum(1)
        control_cost = control_cost.sum(1)

        if self.verbose:
            print('pos_cost: {:5.4f}'.format(pos_cost.mean().detach().cpu().numpy()))
            print('vel_cost: {:5.4f}'.format(vel_cost.mean().detach().cpu().numpy()))
            print('control_cost: {:5.4f}'.format(control_cost.mean().detach().cpu().numpy()))
            print('terminal_cost: {:5.4f}'.format(terminal_cost.mean().detach().cpu().numpy()))
            if cost is not None:
                print('energy_cost: {:5.4f}'.format(energy_cost.mean().detach().cpu().numpy()))
            print('')

        costs = pos_cost + vel_cost + control_cost + terminal_cost + energy_cost
        return costs.view(num_ctrl_samples, num_state_samples)
