from steintorch.distribution.base import Distribution

class Gaussian(Distribution):

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

        self.dim = self.covariance.dim


    def covariance_operator(self, *args):
        raise NotImplementedError
