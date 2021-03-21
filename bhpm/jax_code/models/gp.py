# File: gp.py
# Created Date: 2021-03-12
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Basic vanilla GP to get things going
"""

from functools import partial
import pickle

from jax import random
import jax.numpy as np
import numpy as onp

from ..distributions import mvnorm_logpdf
from ..functions import cholesky, jitchol, trtrs

from .transforms import transform_params


def gp(mean_funcs, kernel_funcs, likelihood_funcs, safe=True):
    """
    Vanilla single-output GP w/ RBF kernel.
    """

    _chol_func = jitchol if safe else cholesky
    t_wrapper = partial(transform_params, transform=np.exp)

    def init_fun(rng, x, y):
        input_shape = (-1, x.shape[1])
        rng_mean, rng_kernel, rng_likelihood = random.split(rng, 3)
        mean_output_shape, mean_params = mean_funcs["init"](rng, input_shape)
        # Reasonable inits for kernel:
        scales = x.max(axis=0) - x.min(axis=0)
        # scales[scales == 0.0] = 1.0
        kernel_output_shape, kernel_params = kernel_funcs["init"](
            rng_kernel, input_shape, scales=scales, variance=y.var()
        )
        _, likelihood_params = likelihood_funcs["init"](
            rng_likelihood, kernel_output_shape, noise=0.001 * y.var()
        )
        params = {
            "mean": mean_params,
            "kernel": kernel_params,
            "likelihood": likelihood_params,
            "x": x,
            "y": y,
        }
        return (input_shape[0],), params

    @t_wrapper
    def gaussian_fun(
        params,
        inputs,
        noise=False,
        diag=True,
        cache=None,
        chol_func=None,
        condition=None,
        return_std=True,
    ):
        """
        Compute the posterior mean & (co)variance

        :param condition: Tuple of (X,y). If provided, condition on this data
        rather than what's stored in params.

        :return:
            diag (std): (N*,), (N*,)
            not  (cov): (N*,), (N*,N*)
        """
        # TODO use cached values
        # NOTE: this doesn't use gp_conditional because we need to Condition
        # *with* the likelihood (jitter)!
        chol_func = _chol_func if chol_func is None else chol_func
        x, y = (params["x"], params["y"]) if condition is None else condition
        mu_x, mu_xs = [mean_funcs["apply"](params["mean"], z) for z in (x, inputs)]
        ckyy = chol_func(
            likelihood_funcs["apply"](
                params["likelihood"],
                None,
                kernel_funcs["apply"](params["kernel"], x, x),
            )[1]
        )
        kfs = kernel_funcs["apply"](params["kernel"], x, inputs)
        a, v = trtrs(ckyy, (y - mu_x)[:, None]), trtrs(ckyy, kfs)

        mean = (a.T @ v).squeeze() + mu_xs
        if diag:
            var = kernel_funcs["apply_diag"](params["kernel"], inputs) - (v * v).sum(
                axis=0
            )
            if noise:
                mean, var = likelihood_funcs["apply"](params["likelihood"], mean, var)
            return mean, (np.sqrt(var) if return_std else var)
        else:
            cov = kernel_funcs["apply"](params["kernel"], inputs, inputs) - v.T @ v
            if noise:
                mean, cov = likelihood_funcs["apply"](params["likelihood"], mean, cov)
            ccov = chol_func(cov)
            return mean, ccov

    # Note: gaussian_fun takes care of transforms
    def apply_fun(
        params, rng, inputs, noise=False, diag=True, cache=None, condition=None
    ):
        """
        Sample GP posterior
        """
        n_test = inputs.shape[0]
        if diag:
            mean, std = gaussian_fun(
                params,
                inputs,
                noise=noise,
                diag=diag,
                cache=cache,
                condition=condition,
            )
            return mean + std * random.normal(rng, shape=(n_test,))
        else:
            mean, ccov = gaussian_fun(
                params,
                inputs,
                noise=noise,
                diag=diag,
                cache=cache,
                condition=condition,
            )
            return mean + (ccov @ random.normal(rng, shape=(n_test, 1))).squeeze()

    @t_wrapper
    def loss_fun(params, data):
        """
        Do NOT use the data stored in params or else optimization will change
        the dataset!
        """
        x, y = data
        fmean = mean_funcs["apply"](params["mean"], x)
        kff = kernel_funcs["apply"](params["kernel"], x, x)
        ymean, kyy = likelihood_funcs["apply"](params["likelihood"], fmean, kff)
        ckyy = cholesky(kyy)
        return -mvnorm_logpdf(y, ymean, ckyy)

    return {
        "init": init_fun,
        "gaussian": gaussian_fun,
        "apply": apply_fun,
        "loss": loss_fun,
    }


def lite(mean_funcs, kernel_funcs):
    """
    Quick & dirty GP posterior mean function to be used for quick & stable
    inference (suitable for single precision!)

    uses the following pre-computed matrices:
    * Mean uses alpha = inv(Kyy) @ (y-mu(x))
    * Variance uses ckyy = chol(Kyy) (Better than precomptued inverse for numerics with
      comparable speed.)
    """

    def _params_to_jax(params):
        if isinstance(params, dict):
            return {key: _params_to_jax(val) for key, val in params.items()}
        elif isinstance(params, list):
            return [_params_to_jax(v) for v in params]
        elif isinstance(params, tuple):
            return tuple([_params_to_jax(v) for v in params])
        elif isinstance(params, onp.ndarray):
            return np.array(params)
        else:
            raise RuntimeError("Unknown type %s" % str(type(params)))

    def init_fun(filename):
        with open(filename, "rb") as f:
            _params = pickle.load(f)
        params = _params_to_jax(_params)
        return params

    def mean(params, rng, inputs):
        """
        Posterior mean @ inputs

        :param rng: Not used; just for consistent API
        :param inputs: (N,DX)

        :return: (N,)
        """
        ksf = kernel_funcs["apply"](params["kernel"], inputs, params["x"])
        return ksf @ params["alpha"][:, 0] + mean_funcs["apply"](params["mean"], inputs)

    def variance(params, inputs, noise=False):
        """
        Compute the marginal posterior variance @ inputs

        :param inputs: (N,DX)

        :return: (N,)
        """
        assert not noise, "For now."
        kss = kernel_funcs["apply_diag"](params["kernel"], inputs)
        kfs = kernel_funcs["apply"](params["kernel"], params["x"], inputs)
        v = trtrs(params["ck"], kfs)
        return kss - np.sum(v * v, axis=0)

    return {"init": init_fun, "apply": mean, "variance": variance}
