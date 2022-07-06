from numpy import diagonal
import torch
from torch.linalg import inv, det, multi_dot

from steintorch.divergence.base import Divergence

l2 = torch.nn.PairwiseDistance(2)  # computes pairwise distances between each component of a tensor
exp_max = 10e30  # clamping parameter for gradient free KSD - prevents numerical issues when target density p(x) is too small


class KSD(Divergence):

    def __init__(self, kernel, preconditioner=torch.Tensor([1])):

        # TODO: Possibly add distribution to __init__ utilise get_score and make score_func optional argument in subsequent code
        # TODO: Different kernels and Stein operators, currently just IMQ with Langevin.

        self.preconditioner = preconditioner

        if kernel is not None:
            if kernel.is_stein is False:
                raise ValueError("Given kernel object is not a Stein kernel")
        self.kernel = kernel

    def eval(self, sample, score, weights=None, preconditioner=None, beta=0.5, V_statistic=True):
        N, d = sample.size()  # number of samples, dimension
        stein_mat = self.stein_matrix(sample=sample, score=score, preconditioner=preconditioner, beta=beta)

        if weights is None:
            if V_statistic is False:
                return 1 / (N * (N - 1)) * (torch.sum(stein_mat) - torch.sum(torch.diag(stein_mat)))
            else:
                return torch.mean(stein_mat)
        else:
            weights = weights / sum(weights)  # normalise the weights
            if V_statistic is False:
                quad_form = multi_dot((weights.unsqueeze(0), stein_mat, weights.unsqueeze(1))).flatten()
                diagonal_terms = torch.dot(weights.pow(2), torch.diag(stein_mat))
                return quad_form - diagonal_terms
            else:
                quad_form = multi_dot((weights.unsqueeze(0), stein_mat, weights.unsqueeze(1))).flatten()
                return quad_form

    def stein_matrix(self, sample, score, preconditioner=None, beta=0.5, no_grad=False):
        N, d = sample.size()  # number of samples, dimension

        if preconditioner is None:
            precon = self.preconditioner
        else:
            precon = preconditioner

        if type(score) is torch.Tensor:
            scores = score
        else:
            scores = score(sample)

        if no_grad is True:
            sample = sample.detach()
            scores = scores.detach()

        PRECON_SIZE = precon.size()
        if len(PRECON_SIZE) == 1 and PRECON_SIZE[0] == 1:
            # scalar preconditioner
            score_prods = (scores.unsqueeze(1) * scores)
            dists = torch.cdist(sample, sample).pow(2)
            diffs = sample.unsqueeze(1) - sample
            score_diffs = scores.unsqueeze(1) - scores
            score_diff_prod = torch.bmm(diffs.view(N * N, 1, d), score_diffs.view(N * N, d, 1)).reshape(N, N)

            k = -4 * beta * (beta + 1) * precon.pow(2) * dists / ((1 + dists * precon).pow(beta + 2))
            k_x = 2 * beta * (d * precon + score_diff_prod * precon) / ((1 + dists * precon).pow(beta + 1))
            k_xy = score_prods.sum(dim=2) / ((1 + dists * precon).pow(beta))
            output = (k + k_x + k_xy)

        elif len(PRECON_SIZE) == 2:
            # full matrix preconditioner
            sqrt_precon = torch.linalg.cholesky(preconditioner)
            trace_precon = torch.trace(preconditioner)

            tsample = torch.matmul(sample, sqrt_precon)
            ttsample = torch.matmul(sample, preconditioner)

            score_prods = (scores.unsqueeze(1) * scores)
            dists2 = torch.cdist(ttsample, ttsample).pow(2)
            dists = torch.cdist(tsample, tsample).pow(2)
            diffs = (sample.unsqueeze(1) - sample).reshape(N * N, d)
            tdiffs = torch.matmul(diffs, sqrt_precon)
            score_diffs = (scores.unsqueeze(1) - scores).reshape(N * N, d)
            tscore_diffs = torch.matmul(score_diffs, sqrt_precon)
            score_diff_prod = torch.bmm(tdiffs.view(N * N, 1, d), tscore_diffs.view(N * N, d, 1)).reshape(N, N)

            k = -4 * beta * (beta + 1) * dists2 / ((1 + dists).pow(beta + 2))
            k_x = 2 * beta * (trace_precon + score_diff_prod) / ((1 + dists).pow(beta + 1))
            k_xy = score_prods.sum(dim=2) / ((1 + dists).pow(beta))
            output = (k + k_x + k_xy)
        else:
            # vector precondioner
            raise NotImplementedError

        return output

    def old_stein_matrix(self, sample, score, preconditioner=None, beta=0.5):
        # TODO: Make more efficient somehow - remove the torch.sum???

        N, d = sample.size()  # number of samples, dimension

        if preconditioner is None:
            precon = self.preconditioner
        else:
            precon = preconditioner

        if type(score) is torch.Tensor:
            scores = score
        else:
            scores = score(sample)

        s1 = scores.repeat(1, N).view(N * N, d)
        s2 = scores.repeat(N, 1)

        diffs = (sample.unsqueeze(1) - sample).reshape(N * N, d)
        dists = torch.cdist(sample, sample).flatten()**2

        PRECON_SIZE = precon.size()
        if len(PRECON_SIZE) == 1 and PRECON_SIZE[0] == 1:
            # lengthscale constant

            dists = torch.cdist(sample, sample).flatten().pow(2)
            k = -4 * beta * (beta + 1) * precon.pow(2) * dists / ((1 + precon * dists).pow(beta + 2))
            k_x = 2 * beta * (d * precon + precon * torch.sum((s1 - s2) * diffs, dim=1)) / ((1 + precon * dists).pow(beta + 1))
            k_xy = torch.sum(s1 * s2, dim=1) / ((1 + precon * dists).pow(beta))
            output_vec = (k + k_x + k_xy)

        return output_vec.reshape(N, N)

    def cross_stein_matrix(self, sample1, sample2, score_func, preconditioner=None, beta=0.5):

        N1, d = sample1.size()
        N2, _ = sample2.size()

        s1 = score_func(sample1)
        s2 = score_func(sample2)

        diffs = sample1 - sample2
        dists = l2(sample1, sample2)**2

        k = -4 * beta * (beta + 1) * dists / ((1 + dists).pow(beta + 2))
        k_x = 2 * beta * (d + torch.sum((s1 - s2) * diffs, dim=1)) / ((1 + dists).pow(1 + beta))
        k_xy = torch.sum(s1 * s2, dim=1) / ((1 + dists).pow(beta))

        output_vec = (k + k_x + k_xy)

        return output_vec.reshape(N1, N2)

    def linear_transform_density(self, log_prob, mat):
        """Computes the density of the transformed variable
            $$ Y = MX, $$ 
            where $M$ is an invertible matrix.

        Args:
            log_prob ([function]): [Log density of random variable]
            matrix ([torch.Tensor]): [Transforming matrix]

        Returns:
            [function]: [description]
        """

        log_mat_determinant = det(mat).log()
        mat_inv = inv(mat)

        def t_log_prob(x):
            t_x = torch.matmul(x, mat_inv)
            return log_prob(t_x) - log_mat_determinant

        return t_log_prob


class GradientFreeKSD(KSD):

    def __init__(self, kernel, preconditioner=torch.Tensor([1])):
        super().__init__(kernel, preconditioner)

    def eval(self, sample, p, q, score_q, preconditioner=None, sigma=1, beta=0.5, weights=None, V_statistic=True, clamp_qp=None, clamp_p=None):
        N, _ = sample.size()  # number of samples, dimension

        if clamp_p is not None:

            if type(p) is torch.Tensor:
                new_log_p = torch.clamp(p, min=clamp_p)
            else:
                new_log_p = lambda x: torch.clamp(p(x), min=clamp_p)

            stein_mat = self.stein_matrix(sample=sample,
                                          log_p=new_log_p,
                                          log_q=q,
                                          score_q=score_q,
                                          preconditioner=preconditioner,
                                          sigma=sigma,
                                          beta=beta,
                                          clamp_qp=clamp_qp)
        else:
            stein_mat = self.stein_matrix(sample=sample,
                                          log_p=p,
                                          log_q=q,
                                          score_q=score_q,
                                          preconditioner=preconditioner,
                                          sigma=sigma,
                                          beta=beta,
                                          clamp_qp=clamp_qp)

        if weights is None:
            if V_statistic is False:
                return 1 / (N * (N - 1)) * (torch.sum(stein_mat) - torch.sum(torch.diag(stein_mat)))
            else:
                return torch.mean(stein_mat)
        else:
            weights = weights / sum(weights)  # normalise the weights
            if V_statistic is False:
                quad_form = multi_dot((weights.unsqueeze(0), stein_mat, weights.unsqueeze(1))).flatten()
                diagonal_terms = torch.dot(weights.pow(2), torch.diag(stein_mat))
                return quad_form - diagonal_terms
            else:
                quad_form = multi_dot((weights.unsqueeze(0), stein_mat, weights.unsqueeze(1))).flatten()
                return quad_form

    def stein_matrix(self, sample, log_p, log_q, score_q, preconditioner=None, sigma=1, beta=0.5, clamp_qp=None, no_grad=False):

        if preconditioner is None:
            precon = self.preconditioner
        else:
            precon = preconditioner

        N, d = sample.size()  # number of samples, dimension

        if type(score_q) is torch.Tensor:
            q_scores = score_q
        else:
            q_scores = score_q(sample)

        if type(log_q) is torch.Tensor:
            log_q_sample = log_q
        else:
            log_q_sample = log_q(sample)

        if type(log_p) is torch.Tensor:
            log_p_sample = log_p
        else:
            log_p_sample = log_p(sample)

        if no_grad is True:
            log_p_sample = log_p_sample.detach()
            log_q_sample = log_q_sample.detach()
            q_scores = q_scores.detach()

        score_prods = (q_scores.unsqueeze(1) * q_scores)
        diffs = sample.unsqueeze(1) - sample
        score_diffs = q_scores.unsqueeze(1) - q_scores
        score_diff_prod = torch.bmm(diffs.view(N * N, 1, d), score_diffs.view(N * N, d, 1)).reshape(N, N)
        dists = torch.cdist(sample, sample).pow(2)

        log_q_p = log_q_sample - log_p_sample
        log_qp_diff = log_q_p.unsqueeze(1) + log_q_p

        if clamp_qp is not None:
            clamped_diff = torch.clamp(log_qp_diff, min=-clamp_qp[0], max=clamp_qp[1])
            coeff = clamped_diff.exp()
        else:
            coeff = log_qp_diff.exp()

        PRECON_SIZE = precon.size()
        if len(PRECON_SIZE) == 1 and PRECON_SIZE[0] == 1:
            k = -4 * beta * (beta + 1) * precon.pow(2) * dists / ((sigma**2 + precon * dists).pow(beta + 2))
            k_x = 2 * beta * (d * precon + precon * score_diff_prod) / ((sigma**2 + precon * dists).pow(beta + 1))
            k_xy = score_prods.sum(dim=2) / ((sigma**2 + precon * dists).pow(beta))
            output = coeff * (k + k_x + k_xy)
        else:
            raise NotImplementedError

        return output

    def cross_stein_matrix(self, sample1, sample2, p, q, score_q, preconditioner=None, sigma=1, beta=0.5):

        N1, d = sample1.size()
        N2, _ = sample2.size()

        q_scores1 = score_q(sample1)
        q_scores2 = score_q(sample2)

        p_sample1 = p(sample1)
        p_sample2 = p(sample2)

        q_sample1 = q(sample1)
        q_sample2 = q(sample2)

        diffs = sample1 - sample2
        dists = l2(sample1, sample2)**2

        coeff = q_sample1 * q_sample2 / (p_sample1 * p_sample2)
        k = -4 * beta * (beta + 1) * dists / ((sigma**2 + dists).pow(beta + 2))
        k_x = 2 * beta * (d + torch.sum((q_scores1 - q_scores2) * diffs, dim=1)) / ((sigma**2 + dists).pow(1 + beta))
        k_xy = torch.sum(q_scores1 * q_scores2, dim=1) / ((sigma**2 + dists).pow(beta))

        output_vec = coeff * (k + k_x + k_xy)

        return output_vec.reshape(N1, N2)


class FunctionalKSD(Divergence):

    def __init__(self, kernel, gaussian, T=None, U=None, DU=None):  #
        """
            Defines infinite dimensional KSD.
            kernel - functional stein kernel (functional means domain is function space)
            gaussian - Gaussian reference measure for Gibbs measure
            T - operator part of infinite dimension kernel
            U - term in Gibbs measure
            DU - the Frechet derivative of U
        """

        self.kernel = kernel
        self.gaussian = gaussian

        self.T = T
        self.U = U
        self.DU = DU

    def eval(self, sample1, sample2):
        return self.kernel.functional_KSD(sample1, sample2, self.gaussian, self.T, self.U, self.DU)
