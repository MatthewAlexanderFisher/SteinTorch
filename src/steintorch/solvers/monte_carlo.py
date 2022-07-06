import torch

from steintorch.solvers.base import Solver


class MonteCarlo(Solver):

    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution


    def eval(self, func, n=1000):

        return torch.mean(func(self.distribution.sample((n,))))
