from pyroapi import optim
import torch
import numpy as np
from qpsolvers import solve_qp

from steintorch.solvers.base import Solver

from scipy.sparse import csc_matrix


class QuadFormQPSolver(Solver):
    """
    A wrapper for the qpsolver package for Pytorch. By default, the solver used is Splitting 
    Conic Solver (scs) - https://www.cvxgrp.org/scs. This needs to be installed separately.

    Assumes the quadratic program is of the form:

    $$ \min_{w | \sum w_i = 1, w_i > 0} w^\top K w. $$
    """

    def __init__(self, K, A, b, lb, solver="scs"):
        self.K = np.array(K.double())
        self.A = np.ones(self.K.shape[0])
        self.b = np.array([1])
        self.lb = np.zeros(self.K.shape[0])

        self.sparseK = csc_matrix(self.K)
        self.sparseA = csc_matrix(self.A)
        self.sparseb = csc_matrix(self.b)
        self.sparselb = csc_matrix(self.lb)

        self.solver = solver

    def optimise(self, eps_abs=1e-7, eps_rel=1e-7):

        optim_weight = torch.Tensor(
            solve_qp(self.sparseK,
                     self.sparselb,
                     None,
                     None,
                     self.sparseA,
                     self.sparseb,
                     self.sparselb,
                     None,
                     solver=self.solver,
                     eps_abs=eps_abs,
                     eps_rel=eps_rel))
        return torch.Tensor(optim_weight)
