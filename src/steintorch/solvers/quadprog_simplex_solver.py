import torch
from scipy.optimize import minimize
from torch.autograd import grad

from steintorch.utils import sqrtm


class QuadProgSimplexSolver:
    """
    Solves the following Quadratic Program for positive definite matrix $K$:

    $$ min_w w^T K w $$

    where $w$ lies in the positive simplex: $w \in \{(x_1,\ldots,x_d) | \sum_{i=1}^d x_i = 1, x_i > 0\}$. 
    This is achieved by parameterising the positive simplex by squaring spherical
    coordinates and running a global optimisation (default BFGS) over $\mathbb{R}^{n-1}$.
    """

    def __init__(self, K):
        self.K = K
        self.sqrtK = sqrtm(K)
        self.dim = K.shape[0]

    def param(self, theta):
        cumprod_sines = torch.cumprod(torch.sin(theta), dim=1)
        stacked_sines = torch.column_stack([torch.ones(theta.shape[0]).unsqueeze(1), cumprod_sines])
        cosines = torch.cos(theta)
        stacked_cosines = torch.column_stack([cosines, torch.ones(theta.shape[0]).unsqueeze(1)])
        return (stacked_cosines * stacked_sines).pow(2)

    def objective(self, theta):
        t_theta = torch.matmul(self.param(theta), self.sqrtK)
        return t_theta.norm(dim=1)

    def quad_form(self, theta):
        x = self.param(theta)
        return (torch.matmul(x, self.K) * x).sum(dim=1)

    def quad_form_np(self, theta):
        theta_ = torch.Tensor(theta).unsqueeze(0)
        return self.quad_form(theta_).flatten().numpy()

    def quad_form_np_grad(self, theta):
        theta_ = torch.Tensor(theta).unsqueeze(0).requires_grad_(True)
        obj_theta = self.quad_form(theta_)
        grads = grad(obj_theta.sum(), theta_, create_graph=True)[0]
        return grads.detach().flatten().numpy()

    def objective_np(self, theta):
        theta_ = torch.Tensor(theta).unsqueeze(0)
        return self.objective(theta_).flatten().numpy()

    def objective_np_grad(self, theta):
        theta_ = torch.Tensor(theta).unsqueeze(0).requires_grad_(True)
        obj_theta = self.objective(theta_)
        grads = grad(obj_theta.sum(), theta_, create_graph=True)[0]
        return grads.detach().flatten().numpy()

    def optimise(self, theta_init, return_theta=False):
        optimumum = minimize(self.objective_np, theta_init, method='BFGS', jac=self.objective_np_grad)
        optimumum_theta = torch.Tensor(optimumum.x).unsqueeze(0)
        optimumum_simplex = self.param(optimumum_theta)
        if return_theta is True:
            return optimumum_simplex, optimumum_theta
        else:
            return optimumum_simplex
