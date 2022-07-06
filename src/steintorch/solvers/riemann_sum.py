import torch

from steintorch.solvers.base import Solver

class RiemannSum(Solver):

    def __init__(self, mesh, volume, density_func=None):
        super().__init__()

        self.mesh = mesh
        self.volume = volume
        self.density_func = density_func


    def eval(self, func, *args):

        if self.density_func is None:
            return self.volume * torch.mean(func(self.mesh))

        else:
            return self.volume * torch.mean(func(self.mesh) * self.density_func(self.mesh))
