import torch
import torch.nn as nn
import math
from functools import lru_cache

from steintorch.kernel.base import Kernel


class MaternKernel(Kernel):

    def __init__(self, dim=1, param_dict={"nu": torch.Tensor([1.5]), "amplitude": torch.Tensor([1]), "lengthscale": torch.Tensor([1])}, basis_class=None):
        super().__init__(dim, param_dict, basis_class)

        self.register_parameter(name="nu", param=nn.Parameter(param_dict.get("nu")))
        self.register_parameter(name="amplitude", param=nn.Parameter(param_dict.get("amplitude")))
        self.register_parameter(name="lengthscale", param=nn.Parameter(param_dict.get("lengthscale")))

    @lru_cache
    def eval(self, x1, x2):
        """Returns the cross-covariance matrix of the kernel at given input locations.
        Raises:
            NotImplementedError: [If the smoothness parameter is not 0.5, 1.5 or 2.5, raise error.]
        Returns:
            [torch.Tensor]: [Cross-covariance matrix at given input locations.]
        """

        distance = torch.cdist(x1, x2, 2) / self.lengthscale
        exp_component = torch.exp(-distance)

        if self.nu == torch.Tensor([0.5]):
            constant = 1
        elif self.nu == torch.Tensor([1.5]):
            constant = 1 + distance
        elif self.nu == torch.Tensor([2.5]):
            constant = (1 + distance + distance.pow(2) / 3)  # (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        else:
            raise NotImplementedError("Only implemented for nu = 0.5, 1.5, 2.5")

        return self.amplitude.pow(2) * constant * exp_component

    def spectral_density_1D(self, s):
        """Returns the spectral density of the kernel at given 1D input locations.
        Returns:
            [torch.Tensor]: [Spectral density of the kernel]
        """

        constant = (self.amplitude.pow(2) * 2**self.dim * math.pi**(self.dim / 2) * (2 * self.nu)**self.nu * torch.exp(torch.lgamma(self.nu + self.dim / 2)) /
                    (torch.exp(torch.lgamma(self.nu)) * self.lengthscale.pow(2 * self.nu)))
        exp_component = s.pow(2) + (2 * self.nu) / (self.lengthscale.pow(2))
        exponent = -(self.nu + self.dim / 2)

        return constant * exp_component.pow(exponent)
