import torch
from torch import nn


class Basis(nn.Module):
    def __init__(self):
        super().__init__()
        self.orthornormal = True
        self.multidim = False  # allows for multidimensional input

    def forward(self):
        """Forward the forward computation .
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def eval(self, x, m):
        raise NotImplementedError

    def get_basis_func(self, coeffs, m):
        
        def basis_func(x):
            basis_eval = self.eval(x, m)
            return torch.matmul(basis_eval, coeffs)

        return basis_func

    def inner_product(self, weights1, weights2):
        assert weights1.shape == weights2.shape
        if self.orthornormal is True:
            if weights1.dim == 1:
                return torch.dot(weights1, weights2)
            else:
                return torch.sum(weights1 * weights2, dim=1)
        else:
            raise NotImplementedError("inner_product assumes orthonormal basis system")
            
    def norm(self, weights):
        if self.orthornormal is True:
            if weights.dim == 1:
                return torch.norm(weights)
            else:
                return torch.norm(weights, dim=1)
        else:
            raise NotImplementedError("norm assumes orthonormal basis system")
