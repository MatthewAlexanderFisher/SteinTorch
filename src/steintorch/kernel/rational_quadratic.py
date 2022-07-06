import torch
import torch.nn as nn
from functools import lru_cache

import math

from steintorch.kernel.base import Kernel


class RationalQuadraticKernel(Kernel):

    def __init__(self, dim=1, param_dict={"shape": torch.Tensor([1]), "amplitude": torch.Tensor([1]), "lengthscale": torch.Tensor([1])}, basis_class=None):
        super().__init__(dim, param_dict, basis_class)

        self.register_parameter(name="shape", param=nn.Parameter(param_dict.get("shape")))
        self.register_parameter(name="amplitude", param=nn.Parameter(param_dict.get("amplitude")))
        self.register_parameter(name="lengthscale", param=nn.Parameter(param_dict.get("lengthscale")))

    @lru_cache
    def eval(self, x1, x2):
        """Returns the cross-covariance matrix of the kernel at given input locations.
        Returns:
            [torch.Tensor]: [Cross-covariance matrix at given input locations.]
        """

        if self.lengthscale.dim() == 1:
            distance = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, 2)
        else:
            inv_l = torch.linalg.cholesky(torch.linalg.inv(self.lengthscale))  # TODO: cache this
            distance = torch.cdist(torch.matmul(x1, inv_l), torch.matmul(x2, inv_l), 2)

        return self.amplitude.pow(2) * (1 + distance).pow(-self.shape)

    def spectral_density_1D(self, s):
        """Returns the spectral density of the kernel at given 1D input locations.
        Returns:
            [torch.Tensor]: [Spectral density of the kernel.]
        """

        const = self.amplitude.pow(2) / (self.lengthscale.pow(self.shape) * torch.exp(torch.lgamma(self.shape)))

        return const * s.pow(self.shape - 1) * torch.exp(-s / self.lengthscale)

    def set_shape(self, shape):
        """Set the shape parameter of the kernel.
        Args:
            shape ([torch.Tensor]): [The shape parameter]
        """
        self.shape = shape


class LangevinSteinRQ(RationalQuadraticKernel):

    def __init__(self, dim=1, parameters=None, basis_class=None):
        super().__init__(dim, parameters, basis_class)

    def eval(self, x1, x2, parameters=None):

        raise NotImplementedError


class GradFreeLangevinSteinRQ(RationalQuadraticKernel):

    def __init__(self, dim=1, parameters=None, basis_class=None):
        super().__init__(dim, parameters, basis_class)

    def eval(self, x1, x2, parameters=None):

        raise NotImplementedError


class FunctionalRQ(RationalQuadraticKernel):

    def __init__(self, dim=1, parameters=None, basis_class=None):
        super().__init__(dim=dim, parameters=parameters, basis_class=basis_class)

    def eval(self, x1, x2):
        raise NotImplementedError


class LangevinFunctionalRQ(RationalQuadraticKernel):

    def __init__(self, dim=1, parameters=None, basis_class=None):
        super().__init__(dim=dim, parameters=parameters, basis_class=basis_class)

    def eval(self, x1, x2):
        raise NotImplementedError
