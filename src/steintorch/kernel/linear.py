import torch
import torch.nn as nn
from functools import lru_cache

from steintorch.kernel.base import Kernel


class LinearKernel(Kernel):

    def __init__(self, dim=1, param_dict={"c": torch.Tensor([1000]), "lengthscale": torch.Tensor([1])}, basis_class=None):
        super().__init__(dim, param_dict, basis_class)

        self.register_parameter(name="c", param=nn.Parameter(param_dict.get("c")))
        self.register_parameter(name="lengthscale", param=nn.Parameter(param_dict.get("lengthscale")))

    @lru_cache
    def eval(self, x1, x2, parameters=None, clamp=False):

        if parameters is None:
            parameters = self.parameters

        if self.lengthscale.dim() == 1:
            distance = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, 2)
        else:
            inv_l = torch.linalg.cholesky(torch.linalg.inv(self.lengthscale))  # TODO: cache this
            distance = torch.cdist(torch.matmul(x1, inv_l), torch.matmul(x2, inv_l), 2)

        if clamp is True:
            return torch.clamp(self.c - distance, min=0)
        else:
            return -distance
