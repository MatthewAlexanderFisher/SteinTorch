from torch.distributions import Categorical

from steintorch.distribution.base import Distribution


class Empirical(Distribution):

    def __init__(self, sampling_func, weights):
        """[summary]

        Args:
            sampling_func ([func]): [Function which takes in integers 0 to n-1 and returns corresponding samples]
            weights ([torch.Tensor]): [Weights in empirical distribution]
        """
        super().__init__()

        self.sampling_func = sampling_func
        self.weights = weights


    def sample(self, n):

        if type(n) is int:
            N = (n,)
        else:
            N = n

        categorical = Categorical(self.weights)
        return self.sampling_func(categorical.sample(N))
