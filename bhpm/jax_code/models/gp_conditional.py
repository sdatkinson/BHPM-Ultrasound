# File: gp_core.py
# Created Date: 2019-11-20
# Author: Steven Atkinson (steven@atkinson.mn)

import jax.numpy as np

from ..functions import trtrs, jitchol
from ..util import clip_up


def point(
    params,
    k,
    x,
    y,
    xs,
    diag=False,
    params_cross=None,
    k_cross=None,
    params_input=None,
    k_input=None,
    kdiag_input=None,
    chol_func=None,
    jitter=None,
):
    """
    Compute basic GP conditional p(ys|y) at xs (conditioned on points) assuming
    zero prior mean (apply it outside of this if you've got something)

    :param params: Kernel parameters
    :param k: Kernel function for conditioning points domain
    :param x: conditioning inputs, (N,DX)
    :param y: conditioning outputs, (N,)
    :param xs: inputs, (NS,DX)
    :param diag: If true, compute diagonal (discard correslations across rows of
    xs)
    :param params_cross: parameters for the input-condition cross kernel (if 
    different)
    :param k_cross: functional form for the cross kernel
    :param params_input: parameters for the input-input kernel (if different)
    :param k_input: functional form for the input-input kernel,
    :param kdiag_input: Provide to speed up diagonal kernel on inputs (otherwise
    we compute the whole thing then take np.diag as a fallback!)
    :param chol_func: Function used to comptue Cholesky.  Default is "safe"
    Cholesky that adds jitter on failure, but this doesn't JIT-compile. You can
    Use an "unsafe" Cholesky to go fast, but be careful!

    :return: mean (NS,), cov
        Shape of cov is as follows:
        * If diag and no special kernels are used, then cov is (NS,)
        * If not diag, it's (NS,NS)
        * If you use a multi-output kernel with DY outputs...
            * diag outputs (NS,DY,DY)
            * ...and non-diag is not implemented!
    """

    k_cross = k if k_cross is None else k_cross
    params_cross = params if params_cross is None else params_cross
    k_input = k if k_input is None else k_input
    params_input = params if params_input is None else params_input
    chol_func = jitchol if chol_func is None else chol_func

    kyy = k(params, x, x)
    if jitter is not None:
        kyy = kyy + jitter * np.eye(*kyy.shape)
    ckyy = chol_func(kyy)
    kfs = k_cross(params_cross, x, xs)
    alpha = trtrs(ckyy, kfs)
    beta = trtrs(ckyy, y[:, None])

    # Hacks here because multi-output didn't broadcast intuitively, unfortunately.
    if alpha.ndim == 2:
        mean = alpha.T @ beta
        if diag:
            kss = (
                np.diag(k_input(params_input, xs, xs))
                if kdiag_input is None
                else kdiag_input(params_input, xs)
            )
            cov = kss - np.sum(alpha * alpha, axis=0)
        else:
            kss = k(params_input, xs, xs)
            cov = kss - alpha.T @ alpha
    if alpha.ndim == 3:  # manually broadcast...big mess unfortunately :(
        dim, n_cond, n = alpha.shape
        alpha = alpha.T  # (N*,M,D)
        beta = np.tile(beta[None], (n, 1, 1))  # (N*,M,1)
        # mean: (N*,D,M)@(N*,M,1)=(N*,D,1)
        # Squeeze to (N*,D)
        mean = (alpha.transpose((0, 2, 1)) @ beta).squeeze()
        assert diag, "For now."
        assert kdiag_input is not None
        kss = kdiag_input(params_input, xs)  # (N*,D,D)
        # (N*,D,D) - (N*,D,M)@(N*,M,D) = (N*,D,D)...whew!
        cov = kss - (alpha.transpose((0, 2, 1)) @ alpha)

    return mean, cov


def marginal(
    k,
    params,
    x,
    y_mu,
    y_ccov,
    xs,
    diag=False,
    k_cross=None,
    kdiag_input=None,
    ckff=None,
    chol_func=None,
    jitter=None,
):
    """
    Marginal GP conditional.
    Pulled from gptorch.

    int p(ys|y)p(y)dy
    """
    chol_func = jitchol if chol_func is None else chol_func
    k_cross = k if k_cross is None else k_cross

    kff = k(params, x, x)
    if jitter is not None:
        kff = kff + jitter * np.mean(np.diag(kff)) * np.eye(*kff.shape)
    ckff = chol_func(kff) if ckff is None else ckff
    kfs = k_cross(params, x, xs)
    alpha = trtrs(ckff, kfs).T
    mean = alpha @ trtrs(ckff, y_mu[:, None])

    # beta @ beta' = inv(L) @ S @ inv(L'), S=cov of conditioning outputs
    beta = trtrs(ckff, y_ccov)
    # gamma @ gamma' = ksf @ inv(Kff) @ S @ inv(Kff) @ kfs
    gamma = alpha @ beta
    if diag:
        kss_diag = (
            np.diag(k(params, xs, xs))
            if kdiag_input is None
            else kdiag_input(params, xs)
        )
        cov = kss_diag - np.sum(alpha * alpha, axis=1) + np.sum(gamma * gamma, axis=1)
    else:
        cov = k(params, xs, xs) - alpha @ alpha.T + gamma @ gamma.T

    return mean, cov
