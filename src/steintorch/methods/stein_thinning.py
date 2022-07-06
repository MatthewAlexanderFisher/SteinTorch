from steintorch.methods.base import Method


class KernelSteinThinning(Method):

    def __init__(self, ksd):

        self.kernel = ksd.kernel
