import torch
"""
Implementations of different linear data standardisers used for KSD methods.
"""


class Standardiser:

    def __init__(self):
        pass

    def compute_standardisation(self, samples):
        raise NotImplementedError

    def t_samples(self, samples):
        raise NotImplementedError

    def t_log_probs(self, log_probs):
        raise NotImplementedError

    def t_scores(self, scores):
        raise NotImplementedError


class StandardiseStdConst(Standardiser):

    def compute_standardisation(self, samples):
        """
        Computes the standardisation $\left(\sum_{i=1}^n \| x_i - \\bar{x} \|\\right^{-1/2} $, where
        $\\bar{x}$ is the mean vector of samples $x_1,\ldots,x_n$.
        """
        return 1 / torch.cdist(samples, samples.mean(dim=0).unsqueeze(0)).pow(2).mean().sqrt().unsqueeze(0)

    def t_samples(self, samples):
        std_ = self.compute_standardisation(samples)
        return samples * std_

    def t_log_probs(self, log_probs):
        std_ = self.compute_standardisation(samples)
        add_const = torch.log(std)


class StandardiseStdVec(Standardiser):

    def compute_standardisation(self, samples):
        """
        Computes the standardisation $ diag(C)^{-1/2} $, where $C$ is the 
        covariance matrix of samples.
        """
        standard_deviations = torch.std(samples, dim=0)
        return 1 / standard_deviations


class StandardiseCovariance(Standardiser):

    def compute_standardisation(self, samples):
        """
        Computes the standardisation $ (C^{-1})^{1/2} $, where $C$ is the 
        covariance matrix of samples.
        """
        if samples.shape[1] == 1:
            return (1 / samples.std()).reshape(1, 1)
        else:
            whitening_mat = torch.linalg.cholesky(torch.linalg.inv(torch.cov(samples.T)))
        return whitening_mat


def standard_std_const(samples):
    """
    Computes the standardisation $\left(\sum_{i=1}^n \| x_i - \\bar{x} \|\\right^{-1/2} $, where
    $\\bar{x}$ is the mean vector of samples $x_1,\ldots,x_n$.
    """
    return 1 / torch.cdist(samples, samples.mean(dim=0).unsqueeze(0)).pow(2).mean().sqrt().unsqueeze(0)


def standard_std_vec(samples):
    """
    Computes the standardisation $ diag(C)^{-1/2} $, where $C$ is the 
    covariance matrix of samples.
    """
    standard_deviations = torch.std(samples, dim=0)
    return 1 / standard_deviations


def standard_cov(samples):
    """
    Computes the standardisation $ (C^{-1})^{1/2} $, where $C$ is the 
    covariance matrix of samples.
    """
    if samples.shape[1] == 1:
        return (1 / samples.std()).reshape(1, 1)
    else:
        whitening_mat = torch.linalg.cholesky(torch.linalg.inv(torch.cov(samples.T)))
    return whitening_mat
