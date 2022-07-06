# Assortment of helper functions

import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg


def multi_deviation(samples, MAD=False):
    """Computes either the (multivariate) standard deviation or 
       mean absolute deviation (MAD). This is non-batched.
    Args:
        samples (torch.Tensor): Samples to compute deviation 
        MAD (bool, optional): Return Mean Absolute Deviation. Defaults to False.

    Returns:
        torch.Tensor: Computed standard deviation or MAD
    """
    # MAD is mean absolute deviation
    mean_vec = samples.T.mean(dim=1).unsqueeze(0)
    distances = torch.cdist(samples, mean_vec)

    if MAD is True:
        return torch.mean(distances)
    else:
        return torch.sqrt(torch.mean(distances.pow(2)))


def nearestSPD(mat):
    """ Calculates nearest semi-positive definite matrix w.r.t. the Frobenius norm (algorithm based on Nick Higham's
    "Computing the nearest correlation matrix - a problem from finance".
    Args:
        mat ([torch.Tensor]): [Input matrix.]
    Returns:
        [torch.Tensor]: [The symmetric positive definite matrix that is closest to input in Frobenius norm.]
    """

    B = (mat + mat.T) / 2
    _, s, V = torch.svd(B)
    H = torch.matmul(V, V.mul(s).T)

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    I = torch.eye(mat.shape[0])
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.symeig(A3)[0].T[0])
        A3 += I * (-mineig * k**2)
        k += 1
    if torch.norm(mat - A3) / torch.norm(A3) > 10:
        print(
            "Matrix failed to be positive definite, distance in Frobenius norm: ",
            torch.norm(mat - A3, p="fro") / torch.norm(A3, p="fro"),
        )
    return A3


def isPD(B):
    """Check whether a matrix is positive definite.
    Args:
        B ([torch.Tensor]): [Input matrix.]
    Returns:
        [bool]: [Returns True if matrix is positive definite, otherwise False.]
    """
    try:
        _ = torch.cholesky(B)
        return True
    except RuntimeError:
        return False


class MatrixSquareRoot(Function):
    """ 
    A differentiable square root of a positive definite matrix. 
    NOTE: matrix square root is not differentiable for matrices with zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        """Compute the square root of the input.
        Args:
            ctx ([type]): [Autograd context]
            input ([type]): [description]
        Returns:
            [torch.Tensor]: [description]
        """
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply
