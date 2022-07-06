import torch

from steintorch.operator.base import Operator


class InnerProduct(Operator):

    """ 
    Computes inner products between functions - 
    
    """

    def __init__(self, solver=None):
        super().__init__()
        self.solver = solver

    def __call__(self, func1, func2, *args):
        
        def prod_func(x):
            return func1(x) * func2(x)

        return self.solver.eval(prod_func, *args)
