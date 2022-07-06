import torch
import math

from steintorch.distribution.gaussian import Gaussian

class BasisGP(Gaussian):

    pass


class LaplaceGP(Gaussian):

    def __init__(self, covariance):

        super().__init__(None, covariance)

        self.domain = torch.Tensor([[0, 1]] * self.dim)

    
    def resampling_func(self, categorical_sample, gp_sample):
        """Returns a categorical resample from a given gp sample

        Args:
            categorical_sample ([torch.Tensor]): [A tensor of samples from a categorical distribution]
        """

        def resampled_func(x):
            result = gp_sample(x).T
            return result[categorical_sample].T

        return resampled_func


    def resampling_coefficient_func(self, categorical_sample, coeff_sample):
        """Returns a categorical resample from a sample of basis coefficients of a given gp sample

        Args:
            categorical_sample ([torch.Tensor]): [A tensor of samples from a categorical distribution]
            coeff_sample ([torch.Tensor]): [A tensor of samples of coefficients from the gp]

        Returns:
            [torch.Tensor]: [A tensor of resampled coefficients]
        """

        return coeff_sample.T[categorical_sample].T


    def cross_covariance(self, x1, x2, m, parameters=None):
        """Compute the cross-covariance matrix at input locations x1 and x2
        Args:
            x1 ([torch.Tensor]): [First Tensor of input locations]
            x2 ([torch.Tensor]): [Second Tensor of input locations]
            m ([int, list]): [Number of basis functions]
            parameters ([list], optional): [Hyperparameters to use in computation]. Defaults to None.
        Returns:
            [type]: [description]
        """
        s_eig = self.covariance.spectral_eig(m, parameters)
        gram_mat_x1 = self.basis_matrix(x1, m).mul(s_eig)
        gram_mat_x2 = self.basis_matrix(x2, m)
        return torch.mm(gram_mat_x1, gram_mat_x2.t())

    def basis_matrix(self, x, m):
        """Compute the basis matrix at given locations x.
        Args:
            x ([torch.Tensor]): [Input locations]
            m ([list, int]): [Number of basis functions]
        Returns:
            [torch.Tensor]: [Matrix of basis function evaluations]
        """
        return self.covariance.basis_eval(x, m)

    def sample_coefficients(self, n, m, random_sample=None):
        M = self.covariance.get_n_basis_funcs(m)

        s_eig = self.covariance.spectral_eig(m)
        s_cov = torch.diag(s_eig).sqrt()

        if random_sample is None:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(torch.Size([M]) + torch.Size([n])).squeeze()
        else:
            sn_samples = random_sample

        mn_samples = torch.matmul(
            s_cov, sn_samples
        ) 

        return mn_samples  

    def sample_prior_from_coefficients(self, coefficients, m=None):
        if m is None:
            m, n = coefficients.shape

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval, coefficients)

        return sampled_function

    def covariance_operator(self, m):
        return self.get_covariance_matrix(m)

    def get_covariance_matrix(self, m):

        M = self.covariance.get_n_basis_funcs(m)

        s_eig = self.covariance.spectral_eig(m)
        cov = torch.diag(s_eig)
        return cov


    def sample_prior(self, n, m, random_sample=None):
        """Sample from the prior GP
        Args:
            n ([int]): [Number of samples]
            m ([list, int]): [Number of basis functions]
            random_sample ([torch.Tensor], optional): [An optional Tensor of samples from the standard Gaussian]. Defaults to None.
        Returns:
            [function]: [Prior samples]
        """

        M = self.covariance.get_n_basis_funcs(m)
        mn_samples = self.sample_coefficients(n, m, random_sample=random_sample)

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval,mn_samples)

        return sampled_function

    def set_kernel_parameters(self, parameters):
        """Set the parameters of the kernel.
        Args:
            parameters ([list]): [List of kernel parameters]
        """
        self.covariance.set_parameters(parameters)

    def set_domain(self, domain):
        """Set the domain of the GP. For example, set_domain([[-1,1],[0,1]]) sets the domain of the GP as the set of points (x,y) satisfying -1 < x < 1, 0 < y < 1.
        Args:
            domain ([list, torch.Tensor]): [List or Tensor defining domain of GP]
        """

        if type(domain) is list:
            domain_t = torch.Tensor(domain)
        else:
            domain_t = domain

        self.domain = domain_t
        self.mean.set_domain(domain_t)
        self.covariance.set_domain(domain_t)

    def _check_dim(self, x):
        """Check that the dimension of x matches the dimension of the kernel.
        Args:
            x ([torch.Tensor]): [Tensor to check the dimension of the kernel against]
        Raises:
            ValueError: [Raises error if dimensions don't match]
        """
        x_dim = x.shape[1]
        if x_dim != self.covariance.dim:
            raise ValueError("Dimension of input doesn't match dimension of kernel.")


