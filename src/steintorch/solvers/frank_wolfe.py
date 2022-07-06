import torch
from torch.autograd import grad

from steintorch.solvers.base import Solver


class FrankeWolfe(Solver):
    """
    PyTorch implementation of the Frank-Wolfe algorithm
    """

    def __init__(self, constraint, objective=None, minimise=None):
        super().__init__()

        self.objective = objective
        self.constraint = constraint

        if minimise is None:
            self.minimise = True
        else:
            self.minimise = False

    def optimise(self, x_init, step="sublinear", lipschitz=None, max_iter=400, tol=1e-12, eps=1e-8, objective=None):
        """Frank-Wolfe algorithm.

        """
        x = x_init.clone().detach().requires_grad_(True)

        if objective is None and self.objective is None:
            raise ValueError("No objective function supplied")
        elif objective is None:
            objective_ = self.objective
        else:
            objective_ = objective

        lipschitz_t = None
        step_size = None
        if lipschitz is not None:
            lipschitz_t = lipschitz

        f_t = objective_(x)
        eval_grad = grad(objective_(x), x)[0]
        old_f_t = None

        for it in range(max_iter):
            update_direction, fw_vertex_rep, away_vertex_rep, max_step_size = self.constraint.oracle(-eval_grad, x, None)
            norm_update_direction = torch.norm(update_direction)**2
            certificate = torch.dot(update_direction, -eval_grad)

            # Compute an initial estimate for the Lipschitz if not given
            if lipschitz_t is None:
                eps = 1e-3
                x_eps = (x + eps * update_direction).detach().requires_grad_(True)
                grad_eps = grad(objective_(x_eps), x_eps)[0]
                lipschitz_t = torch.norm(eval_grad - grad_eps) / (eps * torch.sqrt(norm_update_direction))

            if certificate <= tol:
                break

            if step == "DR":

                if lipschitz is None:
                    raise ValueError('lipschitz needs to be specified with step="DR"')
                step_size = min(certificate / (norm_update_direction * lipschitz_t), max_step_size).detach()
                x_next = (x + step_size * update_direction).detach().requires_grad_(True)
                f_next = objective_(x_next)
                grad_next = grad(objective_(x_next), x_next)[0]

            elif step == "sublinear":
                #  The sublinear 2/(k+2) step-size
                step_size = 2.0 / (it + 2)
                x_next = (x + step_size * update_direction).detach().requires_grad_(True)
                f_next = objective_(x_next)
                grad_next = grad(objective_(x_next), x_next)[0]

            else:
                raise ValueError("Invalid option step=%s" % step)

            x = x + step_size * update_direction
            old_f_t = f_t
            f_t, eval_grad = f_next, grad_next

        return x, it, certificate
