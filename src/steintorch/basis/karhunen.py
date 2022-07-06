import torch
from functools import lru_cache
import math

from steintorch.basis.base import Basis
from steintorch.utils.summation_tensor_gen import sum_tensor_gen


### Implementation of various Karhunen-Loeve expansions

class BrownianBasis(Basis):

    def __init__(self, domain=None):
        super().__init__()

        if domain is None:
            self.domain = [0,1]
        else:
            self.domain = domain

        self.lower, self.upper = self.domain

    @lru_cache(maxsize=16)
    def eval(self, x, m):
        """
        Computes the tensor with Phi_ij = phi_j(x_i). m is the number of basis functions.
        :param x: Tensor of size n x d
        :return: Tensor of size n x m
        """
        
        x_t = (x-self.lower) / (self.upper-self.lower)
        M = math.pi * (torch.linspace(1, m, m) - 0.5)
        
        return math.sqrt(2) * torch.sin(M * x_t) / M


class BrownianBridgeBasis(Basis):

    def __init__(self, domain=None):
        super().__init__()

        if domain is None:
            self.domain = [0,1]
        else:
            self.domain = domain

        self.lower, self.upper = self.domain

    @lru_cache(maxsize=16)
    def eval(self, x, m):
        """
        Computes the tensor with Phi_ij = phi_j(x_i). m is the number of basis functions.
        :param x: Tensor of size n x d
        :return: Tensor of size n x m
        """
        
        x_t = (x-self.lower) / (self.upper-self.lower)
        M = math.pi * torch.linspace(1, m, m)
        
        return math.sqrt(2) * torch.sin(M * x_t) / M
