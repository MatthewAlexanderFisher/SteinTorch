# Implementation of basic kernel density estimation. Constructs a distribution object with evaluatable log-likelihood. Advanced features such as nearest-neighbour methodology are not implemented.

import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Normal, Independent

from steintorch.distribution.base import Distribution
from steintorch.utils.helper import multi_deviation
from steintorch.kernel import SquaredExponentialKernel


class KernelDensityEstimator(Distribution):

    def __init__(self, samples, kernel):
        super().__init__()
        self.samples = samples
        self.kernel = kernel

    def silvermans_rule(self, samples=None, set=True, full_cov=True):
        """Computes a rule-of-thumb lengthscale using Silverman's rule.

        Args:
            samples (torch.Tensor, optional): Samples used to compute Silverman's rule. Defaults to None.


        Returns:
            torch.Tensor: Rule-of-thumb lengthscale
        """
        if samples is None:
            samples = self.samples
        n, d = samples.shape

        if full_cov is True and d > 1:
            sqrt_cov = torch.linalg.cholesky(torch.cov(samples.T, correction=0))
        else:
            sqrt_cov = torch.std(samples, dim=0)

        h = (n * (d + 2) / 4) ** (-1 / (d+4))
        lengthscale = h * sqrt_cov

        if set is True:
            self.kernel.update_parameters({"lengthscale": lengthscale})

        return lengthscale

    def scotts_rule(self, samples=None, set=True):
        """Computes a rule-of-thumb lengthscale using Scott's rule.

        Args:
            samples (torch.Tensor, optional): Samples used to compute Silverman's rule. Defaults to None.


        Returns:
            torch.Tensor: Rule-of-thumb lengthscale
        """
        if samples is None:
            samples = self.samples
        n, d = samples.shape

        standard_deviations = torch.std(samples, dim=0)
        h = torch.Tensor([n ** (-1 / (d+4)) ])
        lengthscale = h * standard_deviations

        if set is True:
            self.kernel.update_parameters({"lengthscale": lengthscale})

        return lengthscale

    def log_prob(self, x):
        
        kernel_eval = self.kernel.normalised_eval(x, self.samples)

        return kernel_eval.mean(dim=1).log()

    def get_cov_matrix(self):
        n, d = self.samples.shape

        sqrt_cov = self.kernel.lengthscale

        if sqrt_cov.dim() == 1 and sqrt_cov.size(0) == 1:
            cov = torch.eye(d) * sqrt_cov.pow(2)
        elif sqrt_cov.dim() == 1:
            cov = torch.diag(sqrt_cov).pow(2)
        else:
            cov = torch.matmul(sqrt_cov, sqrt_cov.T) # assuming cholesky decomp
        return cov

    def sample(self, N):
        n, d = self.samples.shape
        weights = torch.ones(self.samples.size(0))
        mean_vec = self.samples
        
        if type(self.kernel) is SquaredExponentialKernel:
            cov_vec = self.get_cov_matrix().repeat(n,1,1)

            if d == 1:
                mix = Categorical(weights)
                normal = Normal(mean_vec, cov_vec)
                comp = Independent(normal, 1)
                P = MixtureSameFamily(mix, comp)  # PyTorch mixture of univariate normal distributions
            else:
                mix = Categorical(weights)
                P = MixtureSameFamily(mix, MultivariateNormal(mean_vec, cov_vec))

            return P.sample(N)
        else:
            raise NotImplementedError("Chosen kernel cannot be sampled from")
