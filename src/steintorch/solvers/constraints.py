import torch 

class Constraint:

    def __init__(self):
        pass

    def project(self, x, *args):
        raise NotImplementedError

    def oracle(self, u, x, *args):
        raise NotImplementedError

class SimplexConstraint(Constraint):
    def __init__(self, s=None):
        super().__init__()
        
        if s is None:
            self.s = 1
        elif s > 0:
            self.s = s
        else:
            raise ValueError("Radius of simplex must be positive")

    def project(self, x, *args):
        n = x.shape[0]

        # check if we are already on the simplex
        if x.sum() == self.s and torch.all(x >= 0):
            return x

        # get the array of cumulative sums of a sorted (decreasing) copy of x
        u = torch.sort(x,descending=True)[0]
        cssv = torch.cumsum(u, dim=0)
        # get the number of > 0 components of the optimal solution
        rho = torch.nonzero(u * torch.arange(1, n + 1) > (cssv - self.s)).flatten()[-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - self.s) / (rho + 1.0)
        # compute the projection by thresholding x using theta
        w = (x - theta).clip(min=0)
        return w

    def oracle(self, u, x, *args):
        """Return v - x, s solving the linear problem
            max_{||v||_1 = s, v >= 0} <u, v>
        """

        largest_coordinate = torch.argmax(u)

        update_direction = -x.clone()
        update_direction[largest_coordinate] += self.s 

        return update_direction, int(largest_coordinate), None, 1
