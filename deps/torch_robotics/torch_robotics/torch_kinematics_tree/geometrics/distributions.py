import torch
import numpy as np


class Gaussian(object):
    """
    Batch implementation of Gaussian distribution defined on a manifold.
    """

    def __init__(self, manifold, mean, cov):
        self._manifold = manifold
        if mean.shape[0] != manifold.dim_M:
            raise TypeError(f'[Gaussian]: mean dim_M {mean.shape[0]} and the manifold dim_M {manifold.dim_M} mismatch!')
        if cov.shape[0] != manifold.dim_T or cov.shape[1] != manifold.dim_T:
            raise TypeError(f'[Gaussian]: cov shape {cov.shape} and the manifold dim_T {manifold.dim_T, manifold.dim_T} mismatch!')
        self.mean = mean
        self.cov = cov

    def pdf(self, x):
        """
        Evaluates pdf of the distribution at the point x.
        Parameters
        ----------
        :param x: torch of shape (..., self.manifold.dim_M)
        Returns
        -------
        :return pdf_p: torch of shape (...,)
        """
        v = self.manifold.log_map(x, base=self.mean)
        return self._nf * torch.exp(-(v.transpose(-2, -1) @ self.cov_inv @ v).sum(-1) / 2)

    def transform(self, A, b):
        """
        Gaussian transformation.
        :param A: torch of shape (..., dim_T, dim_T), rotation in tangent space
        :param b: np.array of shape (..., dim_M), translation in manifold space
        Returns
        -------
        :return np array of the transformed Gaussians with shape (..., )
        """
        if A.shape[-2] != self.manifold.dim_T or A.shape[-1] != self.manifold.dim_T:
            raise RuntimeError('[Gaussian]: Expected A to be of the dimension of the tangent'
                               ' space (%s, %s) and not %s' % (self.manifold.dim_T, self.manifold.dim_T, A.shape))
        if b.shape[-1] != self.manifold.dim_M:
            raise RuntimeError('[Gaussian]: Expected b to be of the dimension of the manifold'
                               ' space (%s,) and not %s' % (self.manifold.dim_M, b.shape))
        batch_dim = b.shape[:-1]
        mu_trafo = self.manifold.exp_map(A @ (self.manifold.log_map(self.mean)), base=b).view((-1, self.manifold.dim_M))  # (2.53)
        sigma_trafo = self.manifold.matrix_parallel_transport(A @ self.cov @ A.transpose(-2, -1), b, mu_trafo).view((-1, self.manifold.dim_T, self.manifold.dim_T))  # (2.54)
        gaussians = []
        for i in range(mu_trafo.shape[0]):
            gaussians.append(Gaussian(self.manifold, mu_trafo[i], sigma_trafo[i]))
        gaussians = np.array(gaussians).reshape(batch_dim)
        return gaussians

    def kl_divergence_mvn(self, g):
        term1 = torch.log(torch.linalg.det(g.cov)) - torch.log(torch.linalg.det(self.cov))
        term2 = torch.trace(torch.linalg.solve(g.cov, self.cov)) - self.manifold.dim_T
        v = g.manifold.log_map(self.mean, base=g.mean)
        term3 = v.transpose(-2, -1) @ g.cov_inv @ v
        return (term1 + term2 + term3) / 2

    def recompute_inv(self):
        self._cov_inv = torch.linalg.inv(self.cov)
        self._nf = 1 / (torch.sqrt((2 * np.pi)**self.manifold.dim_T * torch.linalg.det(self.cov)))

    @property
    def manifold(self):
        return self._manifold

    @property
    def cov(self):
        return self._cov

    @property
    def nf(self):
        return self._nf

    @cov.setter
    def cov(self, value):
        self._cov = value
        self.recompute_inv()

    @property
    def cov_inv(self):
        return self._cov_inv
