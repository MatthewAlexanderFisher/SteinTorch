# Implementations of various different simple methods to obtain an approximation of a target distribution. 
# Each method either uses evaluations or gradient information from the target density or samples from the target distribution.

from scipy.optimize import minimize
import torch
import numpy as np

from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily, Normal, Independent
from torch.autograd.functional import hessian


#from pystein.utils.gmm import GaussianMixture  - THIS DOESN'T WORK FOR 1D
from sklearn.mixture import GaussianMixture
from steintorch.utils.kde import KernelDensityEstimator


# Laplace approximation

def get_laplace_approximation(p, x0 = np.array([0]), mode=None):
    
    """ Implementation of the Laplace Approximation of a distribution p.
        Computes the MLE using a scipy optimise routine, in this case
        Nelder-Mead and computes the covariance matrix via the Hessian.

        This implementation assumes p is an object with PyTorch auto-differentiable
        log-density.

        Args:
            p ([Distribution]): [Distribution object with PyTorch differentiable density].
            x0 (numpy.array, optional): [Initialisation point for scipy optimiser]. Defaults to np.array([0]).
            mode (torch.Tensor, optional): [If mode is provided, computes Hessian at mode]. Defaults to None.
    """
    # TODO: Implement more complex scipy wrapper for gradient based scipy optimise methods?
    if mode is None:
        numpy_log_prob = lambda x: -p.log_prob(torch.Tensor(x)).numpy()
        mean = torch.Tensor(minimize(numpy_log_prob, x0, method="Nelder-Mead", maxiter=5000).x)
    else:
        mean = mode
    precision_matrix = -hessian(p.log_prob, mean)

    if len(mean) == 1:
        Q = Normal(mean, 1 / precision_matrix.flatten().sqrt())
    else:
        Q = MultivariateNormal(mean, precision_matrix=precision_matrix)

    return Q


# Gaussian Mixture Model from samples

def get_gaussian_mixture_approximation(samples, N=5, covariance_type="full", evaluate=None):
    """ Implementation of a Gaussian Mixture density approximation using samples 
        from the target distribution.

    Args:
        samples ([torch.Tensor]): [Tensor of samples from target distribution].
        N (int, optional): [Number of mixture components of Gaussian mixture model]. Defaults to 5.
        n_iter (int, optional): [Number of iterations of optimiser]. Defaults to 10000.
    """
    _, d = samples.size()

    gmm = GaussianMixture(n_components=N, random_state=0, covariance_type=covariance_type).fit(samples.detach().numpy())

    weights = torch.Tensor(gmm.weights_)
    mean_vec = torch.Tensor(gmm.means_)
    cov_vec = torch.Tensor(gmm.covariances_)

    if mean_vec.dim() == 1:
        mix = Categorical(weights)
        normal = Normal(mean_vec, cov_vec)
        comp = Independent(normal, 1)
        P = MixtureSameFamily(mix, comp)  # PyTorch mixture of univariate normal distributions
    else:
        mix = Categorical(weights)
        P = MixtureSameFamily(mix, MultivariateNormal(mean_vec, cov_vec))

    if evaluate is None:
        return P
    elif evaluate == "aic":
        return P, gmm.aic(samples.detach().numpy())
    elif evaluate == "bic":
        return P, gmm.bic(samples.detach().numpy())
    else:
        raise ValueError("Evaluate argument should only equal None, 'aic' or 'bic'")


# Kernel Density Estimation (uses samples)

def get_kernel_density_approximation(samples, kernel):
    """ Implementation of kernel density estimation using samples
        from the target distribution.

    Args:
        samples ([torch.Tensor]): [Tensor of samples from target distribution].
        kernel ([Kernel]): [Kernel object to be used for KDE]
    """
    return KernelDensityEstimator(samples, kernel)
