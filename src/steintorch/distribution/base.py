class Distribution:
    """
    Base class for distributions.
    """

    def __init__(self):
        pass

    def condition(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def log_prob(self):
        raise NotImplementedError

    def set_condition(self, condition_func):
        self.condition = condition_func

    def set_sample(self, sampling_func):
        self.sample = sampling_func

    def cdf(self):
        raise NotImplementedError

    def icdf(self):
        raise NotImplementedError
