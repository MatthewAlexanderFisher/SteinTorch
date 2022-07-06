import torch
from functools import lru_cache
import math

from steintorch.basis.base import Basis
from steintorch.utils.summation_tensor_gen import sum_tensor_gen

class Laplace(Basis):
    def __init__(self, dim, b=math.pi, c=0):
        """
        General form of basis function is of the form
        sin(i(bx + c)) 
        :param dim:
        """

        super().__init__()

        self.multi_dim = True  # allows for multidimensional inputs

        self.dim = dim  # dimension of the basis functions generated

        self.b = torch.ones(self.dim) * b
        self.c = torch.ones(self.dim) * c
        self.norm = (2 * self.b / math.pi).pow(1 / 2)

        self.index_func = None
        self.integrated_index_func = None

        def func(x, i):
            return torch.sin(i * x)

        self.func = func

        def basis_func(x, i):
            x_t = x.unsqueeze(1)
            return self.func(self.b * x_t + self.c, i)

        self.basis_func = basis_func


    @lru_cache(maxsize=32)
    def eval(self, x, m):
        """
        Computes the tensor with Phi_ij = phi_j(x_i). m is the number of basis functions.
        :param x: Tensor of size n x d
        :return: Tensor of size n x m
        """
        n, dim = x.shape

        j_mat = sum_tensor_gen(dim, m)
        return torch.prod( self.norm * self.basis_func(x, j_mat), dim=2).reshape(n, m ** dim)

    def set_domain(self, domain):
        relative_domain = domain[self.dims]
        new_b = math.pi / (relative_domain.T[1] - relative_domain.T[0])
        new_c = -relative_domain.T[0] * new_b
        self.b = new_b
        self.c = new_c
        self.norm = (2 * self.b / math.pi).pow(1 / 2)


    def set_dims(self, dims):
        self.dims = sorted(dims)

    def set_full_dims(self, full_dims):
        self.full_dims = full_dims
