from numpy import diagonal
import torch
from torch.linalg import multi_dot

from steintorch.divergence.base import Divergence


class MMD:

    def __init__(self, kernel=None):

        self.kernel = kernel

    def eval(self, x, y, weights_x=None, weights_y=None):
        if weights_x is None:
            weights_x = torch.ones(x.size(0)) / x.size(0)
        if weights_y is None:
            weights_y = torch.ones(y.size(0)) / y.size(0)

        k_xx = self.kernel.eval(x, x)
        k_xy = self.kernel.eval(x, y)
        k_yy = self.kernel.eval(y, y)

        quad_form_x = multi_dot((weights_x.unsqueeze(0), k_xx, weights_x.unsqueeze(1))).flatten()
        quad_form_xy = multi_dot((weights_x.unsqueeze(0), k_xy, weights_y.unsqueeze(1))).flatten()
        quad_form_y = multi_dot((weights_y.unsqueeze(0), k_yy, weights_y.unsqueeze(1))).flatten()

        output = quad_form_x - 2 * quad_form_xy + quad_form_y
        return output
