import torch
import numpy as np
import logging
from torch_kinematics_tree.geometrics.quaternion import q_log_map, q_exp_map, q_parallel_transport
from torch_kinematics_tree.geometrics.euclidean import e_log_map, e_exp_map, e_parallel_transport
from torch_kinematics_tree.geometrics.distributions import Gaussian


class Manifold(object):
    """
    Riemannian manifolds.
    "Programming by Demonstration on Riemannian Manifolds", M.J.A. Zeestraten, 2018.
    Attributes
    ----------
    self.name, str, the manifold name, e.g. "R^3 x S^3"
    self.dim_M, int, the manifold space dim_M
    self.dim_T, int, the tangent space dim_M, dim_T <= dim_M
    self.log_map, function, the map from tangent space to manifold: p_M = self.log_map(p_T, base=None), base=None is global map
    self.exp_map, function, the map from manifold to tangent space: p_T = self.exp_map(p_M, base=None), base=None is global map
    self.parallel_transport, the parallel transport map: p_hT = self.parallel_transport(p_gT, g, h)
    """
    logger = logging.getLogger(__name__)

    def __init__(self, dim_M, dim_T, log_map, exp_map, parallel_transport, name, tensor_args=None):
        self._dim_M = dim_M
        self._dim_T = dim_T
        self._log_map = log_map
        self._exp_map = exp_map
        self._parallel_transport = parallel_transport
        self._name = name
        if tensor_args is None:
            device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
            tensor_args = {'device': device, 'dtype': torch.float32}
        self.tensor_args = tensor_args

    def cartesian_product(self, other):
        """
        Compute the cartesian product between the current manifold self and another manifold other.
        """
        dim_M = self.dim_M + other.dim_M
        dim_T = self.dim_T + other.dim_T
        name = self.name + " x " + other.name

        def log_map(p, base=None):
            p1 = p[..., :self.dim_M]
            p2 = p[..., self.dim_M:]
            if base is None:
                b1 = None
                b2 = None
            else:
                b1 = base[..., :self.dim_M]
                b2 = base[..., self.dim_M:]
            return torch.cat((self.log_map(p1, b1), other.log_map(p2, b2)), dim=-1)

        def exp_map(p, base=None):
            p1 = p[..., :self.dim_M]
            p2 = p[..., self.dim_M:]
            if base is None:
                b1 = None
                b2 = None
            else:
                b1 = base[..., :self.dim_M]
                b2 = base[..., self.dim_M:]
            return torch.cat((self.exp_map(p1, b1), other.exp_map(p2, b2)), dim=-1)

        def parallel_transport(p, g, h):
            g1 = g[..., :self.dim_M]
            g2 = g[..., self.dim_M:]
            h1 = h[..., :self.dim_M]
            h2 = h[..., self.dim_M:]
            p1 = p[..., :self.dim_M]
            p2 = p[..., self.dim_M:]
            return torch.cat((self.parallel_transport(p1, g1, h1), other.parallel_transport(p2, g2, h2)), dim=-1)

        return Manifold(dim_M, dim_T, log_map, exp_map, parallel_transport, name)

    def matrix_parallel_transport(self, sigma, g, h):
        Lp = torch.linalg.cholesky(sigma).transpose(-2, -1)
        A_g_h_L = self.parallel_transport(Lp, g, h)
        sigma_parallel = A_g_h_L.transpose(-2, -1) @ A_g_h_L
        return sigma_parallel

    def mean(self, points_in_manifold, **kwargs):
        """
        Computes the mean of the points points_in_manifold on the manifold according to the iterative likelihood maximization.
        """
        weights = kwargs.get('weights', None)
        init_mu = kwargs.get('init_mu', None)
        eps = kwargs.get('eps', 1e-5)
        max_iter = kwargs.get('max_iter', 50)
        return_projections = kwargs.get('return_projections', False)
        if isinstance(points_in_manifold, np.ndarray):
            points_in_manifold = torch.from_numpy(points_in_manifold).to(**self.tensor_args)
        if weights is None:
            weights = torch.ones((points_in_manifold.shape[0], 1), **self.tensor_args)  # batch size
        if init_mu is None:
            mu = points_in_manifold[0]
        else:
            mu = init_mu
        eps_of_mu = None
        k = 0
        for k in range(max_iter):
            mu_last = mu
            eps_of_mu = self.log_map(points_in_manifold, base=mu)
            delta = (eps_of_mu * weights).mean(0)
            mu = self.exp_map(delta, base=mu)
            if torch.norm(mu - mu_last) < eps:
                break
        if k == max_iter - 1:
            Manifold.logger.warning(f'Maximum of {max_iter} iterations reached!')
        if return_projections:
            return mu.squeeze(), eps_of_mu.squeeze()
        else:
            return mu.squeeze()

    def normal_distribution(self, points_in_manifold, **kwargs):
        """
        Computes the Manifold Gaussian (mean and covariance) of the points points_in_manifold on the manifold
        Parameters
        ----------
        :param points_in_manifold: list of np.array of shape (dim_M,) or np.array of shape (dim_M, length).
        Optional parameters
        -------------------
        :param weights: np.array of shape (length,) that contains a weight for each point. None for no weights (all 1).
        :param init_mu: np.array of shape (dim_M,), the initial mean for the iterative algorithm.
        :param regularization: float, a regularization factor that prevents singular covariances
        :param eps: float, if the change in norm of mean is less than eps the algorithm stops.
        :param max_iter: int, the maximum number of iterations.
        Returns
        -------
        :return g: ManifoldGaussian, the Gaussian in this manifold.
        """
        weights = kwargs.get('weights', None)
        init_mu = kwargs.get('init_mu', None)
        regularization = kwargs.get('regularization', 1e-5)
        eps = kwargs.get('eps', 1e-5)
        max_iter = kwargs.get('max_iter', 100)
        mu, eps_of_mu = self.mean(
            points_in_manifold,
            weights=weights,
            init_mu=init_mu,
            eps=eps,
            max_iter=max_iter,
            return_projections=True,
        )
        sigma = torch.cov(eps_of_mu.T, aweights=weights) + regularization * torch.eye(self.dim_T, **self.tensor_args)  # Table 2.4
        return Gaussian(self, mu, sigma)

    def gaussian_product(self, g_list, eps=1e-5, max_iter=100):
        """
        Computes the Gaussian product of the ManifoldGaussians specified in g_list with the iterative likelihood maximization.
        Table 2.3, Table 2.4
        Parameters
        ----------
        :param g_list: list of ManifoldGaussian
        Optional parameters
        -------------------
        :param eps: float, if the change in norm of mean is less than eps the algorithm stops.
        :param max_iter: int, the maximum number of iterations.
        Returns
        -------
        :return g: Gaussian, the combined Gaussian in this manifold.
        """
        dim_T = self.dim_T
        mus = torch.stack([g.mean.squeeze() for g in g_list], dim=0)
        mu = self.mean(mus)  # initial guess
        J_T_W = torch.zeros((dim_T, len(g_list) * dim_T), **self.tensor_args)
        W = torch.zeros((len(g_list) * dim_T, len(g_list) * dim_T), **self.tensor_args)
        sigma_inv = torch.zeros((dim_T, dim_T), **self.tensor_args)
        # start
        k = 0
        log_last = -np.inf
        eps_of_mu = torch.zeros(dim_T * len(g_list), **self.tensor_args)
        for p, g in enumerate(g_list):
            eps_of_mu[dim_T * p:dim_T * (p + 1)] = self.log_map(g.mean, base=mu)
        for k in range(max_iter):
            sigma_inv *= 0
            mu_last = mu
            for p, g in enumerate(g_list):
                sigma_parallel_p = self.matrix_parallel_transport(g.cov, g.mean, mu)
                J_T_W[:, dim_T * p:dim_T * (p + 1)] = -torch.linalg.inv(sigma_parallel_p)
                W[dim_T * p:dim_T * (p + 1), dim_T * p:dim_T * (p + 1)] = J_T_W[:, dim_T * p:dim_T * (p + 1)]
                sigma_inv += -J_T_W[:, dim_T * p:dim_T * (p + 1)]
            # backtracking line search
            log_likelihood = -np.inf
            delta = -torch.linalg.solve(sigma_inv, J_T_W @ eps_of_mu)
            s = 1
            while log_likelihood <= log_last and np.abs(s) > 1e-30:
                mu = self.exp_map(delta * s, base=mu_last)
                for p, g in enumerate(g_list):
                    eps_of_mu[dim_T * p:dim_T * (p + 1)] = self.log_map(g.mean, base=mu)
                log_likelihood = eps_of_mu @ W @ eps_of_mu
                s /= -np.sqrt(2)
            if log_likelihood - log_last < eps:
                break
            log_last = log_likelihood
        if k == max_iter - 1:
            Manifold.logger.warning(f'Iterative likelihood maximization for mean terminated after reaching {max_iter} iterations')
        return Gaussian(self, mu.squeeze(), torch.linalg.inv(sigma_inv))

    def get_origin(self):
        spaces = self.name.split(" x ")
        origin = []
        for man_name in spaces:
            if man_name == "S^3":
                origin.extend([1., 0., 0., 0.])
            elif man_name[:2] == "R^":
                n = int(man_name[2:])
                origin.extend([0.] * n)
            else:
                Manifold.logger.error(f'Invalid manifold naming {man_name}.')
        return torch.tensor(origin, **self.tensor_args)

    def get_pos_quat_indices(self, tangent=False):
        spaces = self.name.split(" x ")
        pos_idx, quat_idx = [], []
        i = 0
        for man_name in spaces:
            if man_name == "S^3":
                r = 3 if tangent else 4
                quat_idx.extend(range(i, i + r))
                i += r
            elif man_name[:2] == "R^":
                n = int(man_name[2:])
                pos_idx.extend(range(i, i + n))
                i += n
            else:
                Manifold.logger.error(f'Invalid manifold naming {man_name}.')
        return np.array(pos_idx), np.array(quat_idx)

    @staticmethod
    def get_quaternion_manifold():
        return Manifold(4, 3, q_log_map, q_exp_map, q_parallel_transport, "S^3")

    @staticmethod
    def get_euclidean_manifold(n):
        return Manifold(n, n, e_log_map, e_exp_map, e_parallel_transport, ("R^%s" % n))

    @staticmethod
    def get_manifold_from_name(name):
        manifold_names = name.split(" x ")
        manifolds = []
        for man_name in manifold_names:
            if man_name == "S^3":
                manifolds.append(Manifold.get_quaternion_manifold())
            elif man_name[:2] == "R^":
                n = int(man_name[2:])
                manifolds.append((Manifold.get_euclidean_manifold(n)))
            else:
                Manifold.logger.error(f'Invalid manifold naming {man_name}.')
        full_manifold = manifolds[0]
        for man in manifolds[1:]:
            full_manifold = full_manifold.cartesian_product(man)
        return full_manifold

    @property
    def dim_M(self):  # noqa
        return self._dim_M

    @property
    def dim_T(self):  # noqa
        return self._dim_T

    @property
    def log_map(self):
        return self._log_map

    @property
    def exp_map(self):
        return self._exp_map

    @property
    def parallel_transport(self):
        return self._parallel_transport

    @property
    def name(self):
        return self._name