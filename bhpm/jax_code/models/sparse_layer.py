# File: sparse_layer.py
# Created Date: 2021-03-12
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Sparse GP layer
"""

from collections import namedtuple
from copy import deepcopy
from functools import partial
import pickle

from jax import random
import jax.numpy as np
from jax.scipy.stats import norm

from ..functions import jitchol, jitchol2, cholesky, logdetchol, trtrs

from ..util import kmeans_centers
from . import gp_conditional as conditional
from . import transforms


def param_transform(x):
    if x.ndim == 2 and x.shape[0] == x.shape[1]:
        return transforms.lower_cholesky()["forward"](x)
    else:
        return np.exp(x)


t_wrapper = partial(transforms.transform_params, transform=param_transform)


def sparse_layer(n, mean_funcs, kernel_funcs, likelihood_funcs, safe=True, jitter=None):
    """
    Sparse variational GP layer, where the posterior over the data is
    approximated as the conditional GP marginalized over inducing variables.

    :param n: How many data are in the dataset that the inducing points are
    summarizing. Used to get the correct scale for the ELBO when doing
    minibatches.
    :param jitter: Applied to the conditioning matrix during GP conditional
    computations.
    """

    chol_func = jitchol if safe else cholesky

    def _ensure_inducing(rng, input_shape, xu, q_mu, q_sqrt, kernel_params, m=100):
        if xu is None:
            xu = random.normal(rng, shape=(m, input_shape[1]))
        if q_mu is None:
            # In my convention, inducing variables are w/o the mean function
            # (because you just subtract it anyways when you condition).
            q_mu = np.zeros((xu.shape[0],))
        if q_sqrt is None:
            q_sqrt = 0.1 * chol_func(kernel_funcs["apply"](kernel_params, xu, xu))
        return xu, q_mu, q_sqrt

    @t_wrapper
    def _sample_fu(params, rng, with_mean=False):
        # In my convention, inducing variables are w/o the mean function
        # (because you just subtract it anyways when you condition)
        m, d = params["q_mu"].size, 1
        fu = (
            params["q_mu"]
            + (params["q_sqrt"] @ random.normal(rng, shape=(m, d))).squeeze()
        )
        if with_mean:
            fu = fu + mean_funcs["apply"](params["mean"], params["xu"])
        return fu

    @t_wrapper
    def kl(params):
        # In my convention, inducing variables are w/o the mean function
        # (because you just subtract it anyways when you condition)
        xu, q_mu, q_sqrt = [params[z] for z in ["xu", "q_mu", "q_sqrt"]]
        kuu = kernel_funcs["apply"](params["kernel"], xu, xu)
        if jitter is not None:
            kuu = kuu + jitter * np.mean(np.diag(kuu)) * np.eye(*kuu.shape)
        ckuu = chol_func(kuu)

        assert q_mu.ndim == 1, "For now"

        logdets = 0.5 * (logdetchol(ckuu) - logdetchol(q_sqrt))
        trace = np.power(trtrs(ckuu, q_sqrt), 2).sum()
        quad = np.power(trtrs(ckuu, q_mu), 2).sum()
        const = -0.5 * q_mu.size

        return logdets + trace + quad + const

    def init_fun(rng, input_shape, xu=None, q_mu=None, q_sqrt=None, m=None):
        rng, rng_mean, rng_kernel, rng_likelihood = random.split(rng, num=4)
        mean_output_shape, mean_params = mean_funcs["init"](rng_mean, input_shape)
        kernel_output_shape, kernel_params = kernel_funcs["init"](
            rng_kernel, input_shape
        )
        _, likelihood_params = likelihood_funcs["init"](
            rng_likelihood, kernel_output_shape
        )
        rng, rng_inducing = random.split(rng)
        xu, q_mu, q_sqrt = _ensure_inducing(
            rng_inducing, input_shape, xu, q_mu, q_sqrt, kernel_params, m=m
        )
        params = {
            "mean": mean_params,
            "kernel": kernel_params,
            "likelihood": likelihood_params,
            "xu": xu,
            "q_mu": q_mu,
            "raw_q_sqrt": transforms.lower_cholesky()["inverse"](q_sqrt),
        }
        return (input_shape[0],), params

    @t_wrapper
    def gaussian_fun(
        params,
        rng,
        inputs,
        sample_fu=False,
        noise=False,
        diag=True,
        cache=None,
        condition=None,
    ):
        """
        Compute the posterior mean & (co)variance conditioned on a sample of the
        inducing varaibles (NOT marginalizing!)

        :param sample_fu: If true, draw a sample from the induced outputs'
        posterior and compute the Gaussian conditioned on that sample. If false,
        then marginalize over the induced outputs' (Gaussian) variational
        posterior.
        :param condition: If provided, use these instead of the inducing data.
            NOTE: The induced output mean q_mu is defined to be without the mean
            function. If you provide condition data, we'll take care of removing
            the mean so that conditional.point() (which doesn't get the mean
            function) works correctly.
        """
        # TODO use cached values
        # TODO use non-safe-apply kernel when possible
        if condition is not None:
            sample_fu = True
        if sample_fu:  # Sample the inducing variables and condition on the sample
            if condition is None:
                xu = params["xu"]
                rng, rng_fu = random.split(rng)
                fu = _sample_fu(params, rng_fu)
            else:
                xu, fu = condition
                # Remove the mean function from the provided condition points.
                fu = fu - mean_funcs["apply"](params["mean"], xu)
            mean_nomf, cov = conditional.point(
                params["kernel"],
                kernel_funcs["safe_apply"],
                xu,
                fu,
                inputs,
                diag=diag,
                kdiag_input=kernel_funcs["apply_diag"],
                chol_func=chol_func,
                jitter=jitter,
            )
        else:  # marginalize out the inducing points
            mean_nomf, cov = conditional.marginal(
                kernel_funcs["safe_apply"],
                params["kernel"],
                params["xu"],
                params["q_mu"],
                params["q_sqrt"],
                inputs,
                diag=diag,
                kdiag_input=kernel_funcs["apply_diag"],
                chol_func=chol_func,
                jitter=jitter,
            )
        # Assuming scalar output for now.
        mu_xs = mean_funcs["apply"](params["mean"], inputs)
        mean = mean_nomf.squeeze() + mu_xs

        if noise:
            mean, cov = likelihood_funcs["apply"](params["likelihood"], mean, cov)

        if diag:
            std = np.sqrt(cov)
            return mean, std
        else:
            ccov = chol_func(cov)
            return mean, ccov

    @t_wrapper
    def apply_fun(
        params, rng, inputs, noise=False, diag=True, cache=None, condition=None
    ):
        """
        Sample GP posterior

        :return: (N,)
        """
        n_test = inputs.shape[0]
        rng_gaussian, rng_sample = random.split(rng)
        mean, cov = gaussian_fun(
            params,
            rng_gaussian,
            inputs,
            noise=noise,
            diag=diag,
            cache=cache,
            condition=condition,
        )
        if diag:
            # "cov" is actually stddev!
            return mean + cov * random.normal(rng, shape=(n_test,))
        else:
            # "cov" is actually cholesky of cov!
            return mean + (cov @ random.normal(rng, shape=(n_test, 1))).squeeze()

    def sample_fun_fun(rng, params, extra_condition=None):
        """
        Create a determinsistic function f(x): (N,DX)->(N,) sampled from the posterior.
        TODO Cache everything that's expensive upfront
        """
        if extra_condition is not None:
            raise NotImplementedError("Gotta use cache for this one!")
            params = deepcopy(params)

        def f(x):
            return gaussian_fun(params, rng, x, condition=True)[0]

        return f

    @t_wrapper
    def loss_fun(params, rng, data, cache=None):
        x, y = data
        y = y.squeeze()
        n_batch = x.shape[0]
        loc, scale = gaussian_fun(params, rng, x, noise=True, diag=True, cache=cache)
        return -(n / n_batch * norm.logpdf(y, loc, scale).sum() - kl(params))

    @t_wrapper
    def save_lite(params, rng, filename):
        """
        Compute the necessary params for a lite GP and save as a pickle.
        Use a sample of the inducing variables as the conditioning data.

        Since conditioning is on inducing variables with live in latent output space we
        don't apply the likelihood to Kff.
        """
        xc = params["xu"]
        fc = _sample_fu(params, rng)  # No mean because we're just going to take it away
        kff = kernel_funcs["apply"](params["kernel"], xc, xc)
        if jitter is not None:
            kff = kff + jitter * np.eye(*kff.shape)
        ck = jitchol(kff)  # Cholesky of kernel
        alpha = trtrs(ck.T, trtrs(ck, fc[:, None]), lower=False)  # (N,1)
        params = deepcopy(params)
        params["x"] = params["xu"]
        params["alpha"] = alpha
        params["ck"] = ck
        with open(filename, "wb") as f:
            pickle.dump(params, f)

    return {
        "mean": mean_funcs,
        "kernel": kernel_funcs,
        "likelihood": likelihood_funcs,
        "init": init_fun,
        "gaussian": gaussian_fun,
        "apply": apply_fun,
        "loss": loss_fun,
        "kl": kl,
        "sample_function": sample_fun_fun,
        "sample_fu": _sample_fu,
        "_chol": chol_func,
        "save_lite": save_lite,
    }


def convert_gp_to_sparse_layer(
    rng,
    mean_funcs,
    kernel_funcs,
    likelihood_funcs,
    gp_funcs,
    params,
    data,
    m_max=100,
    **kwargs
) -> dict:
    """
    Get function dict and parameters for a sparse layer by using an existing
    GP as a starting point
    """
    x, y = data
    assert y.ndim == 1
    m = min(y.size, m_max)
    # idx = random.shuffle(rng, np.arange(x.shape[0]))[:m]
    # xu = deepcopy(x[idx])
    xu = deepcopy(x) if m == x.shape[0] else np.array(kmeans_centers(x, m))
    q_mu_mf, q_sqrt = gp_funcs["gaussian"](
        params, xu, noise=False, diag=False, chol_func=jitchol
    )
    # In my convention, inducing variables are w/o the mean function (because
    # you just subtract it anyways when you condition)
    q_mu = q_mu_mf - mean_funcs["apply"](params["mean"], xu)

    params = deepcopy(params)
    params["xu"] = xu
    params["q_mu"] = q_mu
    params["raw_q_sqrt"] = transforms.lower_cholesky()["inverse"](q_sqrt)

    sparse_layer_funcs = sparse_layer(
        x.shape[0], mean_funcs, kernel_funcs, likelihood_funcs, **kwargs
    )
    return sparse_layer_funcs, params
