from abc import ABC, abstractmethod
import torch
import torch.distributions as dist
import numpy as np


class GMM:
    def __init__(
            self,
            means,
            sigmas,
            weights,
    ):
        """
        Parameters
        ----------
        means :
            shape: [num_particles, steps, ctrl_dim]
        sigmas :
            shape: [num_particles, steps, ctrl_dim]
        weights :
            shape: [num_particles]
        """
        self.num_particles, self.rollout_steps, self.ctrl_dim = means.shape
        components = dist.Independent(dist.Normal(means, sigmas), 2)
        mixture = dist.Categorical(weights)
        self.dist = dist.mixture_same_family.MixtureSameFamily(mixture, components)

    def sample(self, num_samples):
        return self.dist.sample(num_samples).transpose(0, 1)

    def log_prob(self, x):
        return self.dist.log_prob(x)


def avg_ctrl_to_goal(
        state,
        target,
        rollout_steps,
        dt,
        max_ctrl=100,
        control_type='velocity',
):
    """ Average control from state to target. Control must be
    first/second-derivative in state/target space. Ex. velocity control for
    cartesian states"""
    assert control_type in [
        'velocity',
        'acceleration',
    ]
    if control_type == 'velocity':
        return (
                (target - state)/ rollout_steps / dt
        ).clamp(max=max_ctrl)
    else:
        state_dim = state.dim()
        pos_dim = int(state_dim / 2)
        return (
                (target[:pos_dim] - state[:pos_dim]) / rollout_steps / dt**2
        ).clamp(max=max_ctrl)


def get_indep_gaussian_prior(
        sigma_init,
        rollout_steps,
        control_dim,
        mu_init=None,
):

    mu = torch.zeros(
        rollout_steps,
        control_dim,
    )
    if mu_init is not None:
        mu[:, :] = mu_init

    sigma = torch.ones(
        rollout_steps,
        control_dim,
    ) * sigma_init

    return dist.Normal(mu, sigma)


def get_multivar_gaussian_prior(
        sigma,
        rollout_steps,
        control_dim,
        Cov_type='indep_ctrl',
        mu_init=None,
        tensor_args=None,
):
    """
    :param sigma: standard deviation on controls
    :param control_dim:
    :param rollout_steps:
    :param Cov_type: Covariance prior type, 'indep_ctrl': diagonal Cov,
    'const_ctrl': const. ctrl Cov.
    :param mu_init:
    :return: distribution with MultivariateNormal for each control dimension
    """
    assert Cov_type in [
        'indep_ctrl',
        'const_ctrl',
    ], 'Invalid type for control prior dist.'

    mu = torch.zeros(
        rollout_steps,
        control_dim,
        **tensor_args,
    )

    if mu_init is not None:
        mu[:, :] = mu_init

    if Cov_type == 'const_ctrl':
        # Const-ctrl covariance
        Cov_gen = const_ctrl_Cov

    elif Cov_type == 'indep_ctrl':
        # Isotropic covariance
        Cov_gen = diag_Cov
    else:
        raise IOError('Cov_type not recognized.')

    Cov = Cov_gen(
        sigma,
        rollout_steps,
        control_dim,
        tensor_args=tensor_args,
    )

    # check_Cov_is_valid(Cov)
    return ControlTrajectoryGaussian(
        rollout_steps,
        control_dim,
        mu,
        Cov,
        tensor_args=tensor_args
    )


def diag_Cov(
        sigma,
        length=None,
        ctrl_dim=None,
        tensor_args=None,
):
    """
      Time-independent diagonal covariance matrix. Assumes independence
      across control dimension.
    """
    Cov = torch.eye(
        length,
        **tensor_args,
    ).unsqueeze(-1).repeat(1, 1, ctrl_dim)

    if isinstance(sigma, list):
        Cov = Cov * torch.Tensor(np.array(sigma)).to(**tensor_args)**2
    else:
        Cov = Cov * sigma**2

    return Cov


def const_ctrl_Cov(
        sigma,
        length=None,
        ctrl_dim=None,
        tensor_args=None,
):
    """
      Constant-control covariance prior. Assumes independence across control
      dimension.
    """

    if isinstance(sigma, list):
        sigma = torch.from_numpy(np.array(sigma)).to(**tensor_args)

    L = torch.tril(
        torch.ones(
            length,
            length-1,
            **tensor_args,
        ), diagonal=-1,
    )
    LL_t = torch.matmul(
        L, L.transpose(0, 1)
    )

    LL_t += torch.ones(
        length,
        length,
        **tensor_args,
    )

    Cov = LL_t.unsqueeze(-1).repeat(1, 1, ctrl_dim) * sigma**2
    return Cov


def check_Cov_is_valid(Cov):
    """
    Check determinant of Cov to pre-empt potential numerical instability
    in Multi-variate Gaussian.
    For example, dist.logprob(x) >> 1.
    :param Cov: covariance matrix (Tensor) of shape [rollout_length,
    rollout_length, control_dim]
    """
    Cov_np = Cov.cpu().numpy()
    for i in range(Cov.shape[-1]):
        det = np.linalg.det(Cov_np[:,:,i])
        if det < 1.e-7:
            raise ZeroDivisionError(
                'Covariance-determinant too small, potential for underflow.  '
                'Consider increasing sigma.'
            )


class ControlTrajectoryPrior(ABC):
    """
    Prior distribution on control trajectories, Assumes independence across
    ctrl_dim.
    """
    def __init__(
            self,
            rollout_steps,
            ctrl_dim,
            tensor_args=None,
    ):
        self.rollout_steps = rollout_steps
        self.ctrl_dim = ctrl_dim
        self.list_ctrl_dists = []  # TODO - make this a tensor, or list of tensors of same distributions...
        if tensor_args is None:
            tensor_args = {'device': 'cpu', 'dtype': torch.float32}
        self.tensor_args = tensor_args

    @abstractmethod
    def make_dist(self):
        """ Construct list of sampling distribution, one for each control
        dimension"""
        pass

    def log_prob(
            self,
            samples,
            cond_inputs=None
    ):
        """
        :param samples: control samples of shape ( num_particles, rollout_steps,
         ctrl_dim)
        :return: log_probs, of shape (num_particles.)
        """
        assert samples.dim() == 3
        assert samples.size(1) == self.rollout_steps
        assert samples.size(2) == self.ctrl_dim
        num_particles = samples.size(0)

        log_probs = torch.zeros(
            num_particles,
            **self.tensor_args,
        )

        for i in range(self.ctrl_dim):
            samp = samples[:, :, i]  # [num_particles, rollout_steps]
            log_probs += self.list_ctrl_dists[i].log_prob(
                samp
            )

        return log_probs

    def update_means(self, means):
        # TODO - remove this for loop after making self.list_ctrl_dists a tensor
        for i in range(self.ctrl_dim):
            self.list_ctrl_dists[i].loc = means[..., i].detach().clone()

    def sample(
            self,
            num_samples,
            cond_inputs=None,
    ):
        """
        :param num_particles: number of control particles
        :param cond_inputs: conditional input (not implemented)
        :return: control tensor, of size (rollout_steps, num_particles,
        ctrl_dim)
        """
        U_s = torch.empty(
            num_samples,
            self.rollout_steps,
            self.ctrl_dim,
            **self.tensor_args
        )
        # TODO - remove this for loop
        for i in range(self.ctrl_dim):
            U_s[:, :, i] = self.list_ctrl_dists[i].sample(
                (num_samples,)
            ).to(**self.tensor_args)
        return U_s


class ControlTrajectoryGaussian(ControlTrajectoryPrior):
    """
    Multivariate Gaussian distribution for each control dimension.
    """
    def __init__(
            self,
            rollout_steps,
            ctrl_dim,
            mu=None,
            Cov=None,
            tensor_args=None,
    ):

        assert mu.size(0) == rollout_steps
        assert mu.size(1) == ctrl_dim

        super().__init__(
            rollout_steps,
            ctrl_dim,
            tensor_args=tensor_args,
        )
        self.mu = mu
        self.Cov = Cov
        self.list_ctrl_dists = []
        self.make_dist()

    def make_dist(self):
        for i in range(self.ctrl_dim):
            self.list_ctrl_dists.append(
                dist.MultivariateNormal(
                    self.mu[:, i],
                    covariance_matrix=self.Cov[:,:,i]
            ))
