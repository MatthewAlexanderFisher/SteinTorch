import torch

from steintorch.utils.summation_tensor_gen import sum_tensor_gen
from steintorch.basis.laplace import Laplace

import torch.nn as nn


class Kernel(nn.Module):

    def __init__(self, dim, parameter_dict, basis_class=None):

        super(Kernel, self).__init__()

        self.dim = dim

        if basis_class is None:
            self.basis_class = Laplace
        else:
            self.basis_class = basis_class

        self.bases = [self.basis_class(self.dim)]
        self.parameter_dict = parameter_dict  # This is probably unnecessary and is also not updated TODO:remove

        self.stationary = None

        self.is_stein = False  # is kernel a Stein kernel?
        self.is_functional = False  # is kernel a functional kernel - i.e. a kernel with a function input?
        self.is_graph = False  # is kernel a graph kernel?

    def eval(self, x1, x2):
        """Computes cross-covariance matrix at given input locations.
        Args:
            x1 ([torch.Tensor]): [First tensor of input locations.]
            x2 ([torch.Tensor]): [Second tensor of input locations.]
            parameter ([type], optional): [description]. Defaults to None.
        Raises:
            NotImplementedError: [Raised Error]
        """
        raise NotImplementedError

    def basis_eval(self, x, m):
        return self.bases[0].eval(x, m)

    def spectral_density_1D(self, x):

        raise NotImplementedError

    def spectral_eig(self, m):

        dim = self.dim
        coeff = self.bases[0].b

        j_mat = sum_tensor_gen(dim, m)
        eig_val = torch.sum((j_mat * coeff).pow(2), dim=1)
        s_eig = self.spectral_density_1D(torch.sqrt(eig_val))
        return s_eig

    def set_domain(self, domain):
        for i in self.bases:
            i.set_domain(domain)

    def update_parameters(self, parameter_dict):
        with torch.no_grad():
            for name, param in self.state_dict().items():
                if parameter_dict.get(name) is not None:
                    new_param = parameter_dict.get(name)
                    try:
                        param.copy_(new_param)  # Update the parameter
                    except:
                        self.add_parameter(name, parameter_dict.get(name))  # Override parameter if different shape

    def add_parameter(self, name, value):
        self.register_parameter(name=name, param=nn.Parameter(value))

    def get_n_basis_funcs(self, m):
        return int(m**self.dim)

    def set_dim(self, dim):
        self.__init__(dim, self.parameters, self.basis_class)
