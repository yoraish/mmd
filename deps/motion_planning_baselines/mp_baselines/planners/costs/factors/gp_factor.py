import torch


class GPFactor:

    def __init__(
            self,
            dim,
            sigma,
            d_t,
            num_factors,
            tensor_args=None,
            Q_c_inv=None,
    ):
        self.dim = dim
        self.d_t = d_t
        self.tensor_args = tensor_args
        self.state_dim = self.dim * 2  # position and velocity
        self.num_factors = num_factors
        self.idx1 = torch.arange(0, self.num_factors, device=tensor_args['device'])
        self.idx2 = torch.arange(1, self.num_factors+1, device=tensor_args['device'])
        self.phi = self.calc_phi()
        if Q_c_inv is None:
            Q_c_inv = torch.eye(dim, **tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(num_factors, dim, dim, **tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()  # shape: [num_factors, state_dim, state_dim]

        ## Pre-compute constant Jacobians
        self.H1 = self.phi.unsqueeze(0).repeat(self.num_factors, 1, 1)
        self.H2 = -1. * torch.eye(self.state_dim, **self.tensor_args).unsqueeze(0).repeat(
            self.num_factors, 1, 1,
        )

    def calc_phi(self):
        I = torch.eye(self.dim, **self.tensor_args)
        Z = torch.zeros(self.dim, self.dim, **self.tensor_args)
        phi_u = torch.cat((I, self.d_t * I), dim=1)
        phi_l = torch.cat((Z, I), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi

    def calc_Q_inv(self):
        m1 = 12. * (self.d_t ** -3.) * self.Q_c_inv
        m2 = -6. * (self.d_t ** -2.) * self.Q_c_inv
        m3 = 4. * (self.d_t ** -1.) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def get_error(self, x_traj, calc_jacobian=True):
        batch, horizon = x_traj.shape[0], x_traj.shape[1]
        state_1 = torch.index_select(x_traj, 1, self.idx1).unsqueeze(-1)
        state_2 = torch.index_select(x_traj, 1, self.idx2).unsqueeze(-1)
        error = state_2 - self.phi @ state_1

        if calc_jacobian:
            H1 = self.H1
            H2 = self.H2
            # H1 = self.H1.unsqueeze(0).repeat(batch, 1, 1, 1)
            # H2 = self.H2.unsqueeze(0).repeat(batch, 1, 1, 1)
            return error, H1, H2
        else:
            return error
