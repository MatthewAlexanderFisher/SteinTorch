import scipy.stats
from scipy import optimize
from torch.autograd import grad

import torch


"""
Utility functions for various CDFs and a general inverse CDF approach
"""


def get_1D_mixture_cdf(normal, weights):

    """Given a 1D PyTorch mixture of normal distributions, parameterised by the normal PyTorch distribution and weights parameter, return its cdf.

    weights = torch.Tensor([0.1,0.9,0.5])
    mean_vec = torch.Tensor([[0],[0.1],[0.06]])
    var_vec = torch.Tensor([[2],[1],[0.7]])

    mix = D.Categorical(weights)
    normal = D.Normal(mean_vec, var_vec)
    comp = D.Independent(normal, 1)
    P = D.MixtureSameFamily(mix, comp)  # PyTorch mixture of univariate normal distributions

    cdf = get_1D_mixture_cdf(normal, weights)  # the outputted cdf

    """

    normalised_weights = weights / weights.sum()
    def cdf(x):
        normal_cdf = normal.cdf(x)
        return (normal_cdf.T * normalised_weights).sum(dim=1)
    
    return cdf


def get_icdf(cdf, log_prob=None, bracket=[-3,3], n_mesh=20, max_iter=200):

    """ 
    Given a PyTorch distribution cdf function, return an inverse cdf function that utilises the scipy newton inverse solver.
    To get a good initialisation for the solver, we first approximately inverse the cdf using grid search, controlled by the bracket and n_mesh parameters.
    The fprime and fprime2 scipy optimize parameters are computed using automatic differentiation, when appropriate
    TODO: One can easily add more functionality by changing the automatic initialisation protocol.
    """

    lin_space = torch.linspace(bracket[0],bracket[1],n_mesh) # the grid used in the grid-search
    cdf_lin_space = cdf(lin_space) # cdf evaluated on the grid

    if log_prob is None:
        # if the log_prob is not provided, utilise the Newton-Raphson approximation

        def grad_cdf(x, a):
            func = lambda y, a: cdf(y) - torch.Tensor(a)
            X = torch.Tensor(x).requires_grad_(True)
            cdf_eval = func(X, a)
            grads = grad(cdf_eval.sum(), X, create_graph=True)[0]
            return grads.detach().numpy()

        def icdf(x):

            initial = lin_space[torch.argmax(-torch.abs(cdf_lin_space - x.unsqueeze(1)), dim=1)] # get a good initialisation (approximately inverts the function using grid search on lin_space defined above)

            func = lambda y, a: cdf(torch.Tensor(y)).numpy() - a
            vec_res = optimize.newton(func, initial.detach().numpy(), fprime=grad_cdf, args=(x.detach().numpy(),), maxiter=max_iter)
            return vec_res

    else:
        # if the log_prob is provided, utilise the second-order Halley's method
        pdf = lambda y, a: log_prob(torch.Tensor(y).unsqueeze(1)).exp().flatten().numpy()

        def grad_pdf(x, a):
            func = lambda y, a: log_prob(y.unsqueeze(1)).exp().flatten()
            X = torch.Tensor(x).requires_grad_(True)
            pdf_eval = func(X, a)
            grads = grad(pdf_eval.sum(), X, create_graph=True)[0]
            return grads.detach().numpy()

        def icdf(x):
            initial = lin_space[torch.argmax(-torch.abs(cdf_lin_space - x.unsqueeze(1)), dim=1)]
            func = lambda y, a: cdf(torch.Tensor(y)).numpy() - a
            vec_res = optimize.newton(func, initial.detach().numpy(), fprime=pdf, fprime2=grad_pdf, args=(x.detach().numpy(),), maxiter=max_iter)
            return vec_res

    return icdf



def get_scipy_mvn_cdf(mvn):
    """
    PyTorch wrapper for multivariate normal CDF, using scipy. 
    The ```scipy``` implementation which uses a [Fortran implementation](https://github.com/scipy/scipy/blob/master/scipy/stats/mvndst.f) 
    of the cubature scheme (possibly just a Monte-Carlo approximation based on a transformation of the integral) 
    in A. Genz's "[Numerical Computation of Multivariate Normal Probabilities](https://www.jstor.org/stable/1390838)". 
    """
    mean = mvn.mean.numpy()
    cov = mvn.covariance_matrix.numpy()
    scipy_mvn = scipy.stats.multivariate_normal(mean, cov)

    def cdf(x):
        return scipy_mvn.cdf(x.numpy())

    return cdf
