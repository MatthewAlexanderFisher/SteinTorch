from functools import lru_cache
import torch
import torch.nn as nn
from functools import lru_cache

import math

from steintorch.kernel.base import Kernel


class SquaredExponentialKernel(Kernel):

    def __init__(self, dim=1, param_dict={"amplitude": torch.Tensor([1]), "lengthscale": torch.Tensor([1])}, basis_class=None):
        super().__init__(dim, param_dict, basis_class)

        self.register_parameter(name="amplitude", param=nn.Parameter(param_dict.get("amplitude")))
        self.register_parameter(name="lengthscale", param=nn.Parameter(param_dict.get("lengthscale")))

    @lru_cache
    def eval(self, x1, x2):
        if self.lengthscale.dim() == 1:
            distance = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, 2)
        else:
            inv_l = torch.linalg.cholesky(torch.linalg.inv(self.lengthscale))  # TODO: cache this
            distance = torch.cdist(torch.matmul(x1, inv_l), torch.matmul(x2, inv_l), 2)

        return self.amplitude.pow(2) * self.__call__(distance)

    def diff_eval(self, x1, x2):
        if self.lengthscale.dim() == 1:
            distance = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, 2)
        else:
            inv_l = torch.linalg.cholesky(torch.linalg.inv(self.lengthscale))
            distance = torch.cdist(torch.matmul(x1, inv_l), torch.matmul(x2, inv_l), 2)

        pre_term = (self.lengthscale.pow(2) - distance.pow(2)) / self.lengthscale.pow(4)
        return self.amplitude.pow(2) * pre_term * self.__call__(distance)

    def __call__(self, d):
        return torch.exp(-d.pow(2) / 2)

    def normalised_eval(self, x1, x2):
        """Returns the cross-covariance matrix of the kernel at given input locations. The amplitude parameter is set such that the kernel function is a valid density function.

        Returns:
            [torch.Tensor]: [Cross-covariance matrix at given input locations.]
        """

        if self.lengthscale.dim() == 1 and self.lengthscale.size(0) == 1:
            distance = torch.cdist(x1, x2, 2) / self.lengthscale
            determinant = self.lengthscale.pow(self.dim).pow(2)

        elif self.lengthscale.dim() == 1:
            distance = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, 2)
            determinant = self.lengthscale.prod().pow(2)
        else:
            inv_l = torch.linalg.inv(self.lengthscale)
            distance = torch.cdist(torch.matmul(x1, inv_l), torch.matmul(x2, inv_l), 2)
            determinant = torch.linalg.det(self.lengthscale).pow(2)

        constant = (determinant * (2 * math.pi)**self.dim).pow(-0.5)

        return constant * self.__call__(distance)

    def spectral_density_1D(self, s):

        const = self.amplitude.pow(2) * (2 * math.pi * self.lengthscale.pow(2)).pow(1 / 2)
        return const * torch.exp(-(s * self.lengthscale).pow(2) / 2)


class LangevinSteinSE(SquaredExponentialKernel):

    def __init__(self, dim=1, parameters=None, basis_class=None):
        super().__init__(dim, parameters, basis_class)

    def eval(self, x1, x2, parameters=None):

        raise NotImplementedError


class FunctionalSE(SquaredExponentialKernel):

    def __init__(self, T, dim=1, parameters=None, basis_class=None, norm=None):
        super().__init__(dim=dim, parameters=parameters, basis_class=basis_class)
        self.T = T

        if norm is None:
            self.norm = self.bases[0].norm  # use norm from basis class - performs on coefficients of basis expansion
        else:
            self.norm = norm

    def eval(self, x1, x2):
        return torch.exp(-self.norm(self.T(x1) - self.T(x2)).pow(2) / 2)


class LangevinFunctionalSE(FunctionalSE):

    def __init__(self, T, dim=1, parameters=None, basis_class=None, norm=None):
        super().__init__(T, dim=dim, parameters=parameters, basis_class=basis_class, norm=norm)

    def eval(self, x1, x2):
        raise NotImplementedError
