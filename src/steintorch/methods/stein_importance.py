import torch

from steintorch.methods.base import Method
from steintorch.solvers.frank_wolfe import FrankeWolfe
from steintorch.solvers.constraints import SimplexConstraint


class KSDImportanceSampling:

    def __init__(self, ksd, solver=None):
        super().__init__()

        self.ksd = ksd

        if solver is None:
            self.solver = FrankeWolfe(constraint=SimplexConstraint())
        else:
            self.solver = solver

    def eval(self, samples, *args, **kwargs):

        raise NotImplementedError
