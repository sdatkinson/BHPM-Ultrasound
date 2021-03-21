# File: matrix.py
# Created Date: 2020-04-18
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Kernels that are meant to produce matrices of values.
We use the vectorized squared-distance algorithm for speed.
As a trade-off, We avoid doing kernel grads on these.

This is a reasonable place for "modules" to live; kernel grads can be comptued 
by building functions and grabbing the module's parameters
"""

from functools import partial

import jax
import jax.numpy as np

from .transforms import transform_params


def _squared_distance(x1, x2, scales=None):
    z1, z2 = (x1, x2) if scales is None else (x1 / scales, x2 / scales)
    return (  # clip_up(   FIXME
        np.sum(z1 * z1, axis=1, keepdims=True)
        - 2.0 * z1 @ z2.T
        + np.sum(z2 * z2, axis=1, keepdims=True).T
    )


_remat_squared_distance = jax.remat(_squared_distance)


def vmap(k, diag=False):
    """
    Vectorize a "single" kernel of the form k(params, x1, x2)

    diag: k(params, x): (N,DX) -> (N,)
    full: k(params, x1, x2): (N,DX), (M,DX) -> (N,M)
    """
    if diag:
        # k(params, x)
        return jax.vmap(lambda params, x: k(params, x, x), (None, 0))
    else:
        # k(params, x1, x2)
        inside = jax.vmap(lambda params, x1, x2: k(params, x1, x2), (None, None, 0))
        return jax.vmap(lambda params, x1, x2: inside(params, x1, x2), (None, 0, None))
        # Is this faster?
        # return jax.vmap(
        #     lambda params, x1, x2: jax.vmap(
        #         lambda x2: k(params, x1, x2)
        #     )(x2),
        #     (None, 0, None)
        # )


def periodic():
    """
    From Duvenaud, "Automatic model construction with Gaussian processes" (2014)
    Fig. 2.1
    """
    t_wrapper = partial(transform_params, transform=np.exp)

    def init_fun(rng, input_shape):
        params = {
            "raw_variance": np.array(0.0),
            "raw_scale": np.array(0.0),
            "raw_periods": np.zeros((input_shape[1],)),
        }
        return (input_shape[0], input_shape[0]), params

    @t_wrapper
    def apply_fun(params, x1, x2):
        r = np.sqrt(_squared_distance(x1, x2, scales=params["periods"]))
        return params["variance"] * np.exp(
            -params["scale"] * np.power(np.sin(np.pi * r), 2)
        )

    @t_wrapper
    def apply_diag_fun(params, x):
        return params["variance"] * np.ones(x.shape[0])

    @t_wrapper
    def apply_single_fun(params, x1, x2):
        """
        Maps a pair of 1D vectors to a scalar (use this for grads)
        """
        dr = (x1 - x2) / params["periods"]
        r = np.sqrt(np.dot(dr, dr))
        return params["variance"] * np.exp(
            -params["scale"] * np.power(np.sin(np.pi * r), 2)
        )

    return {
        "init": init_fun,
        "apply": apply_fun,
        "apply_diag": apply_diag_fun,
        "apply_single": apply_single_fun,
    }


def rbf():
    t_wrapper = partial(transform_params, transform=np.exp)

    def init_fun(rng, input_shape, scales=None, variance=None):
        params = {
            "raw_variance": np.log(variance) if variance is not None else np.array(0.0),
            "raw_scales": np.log(scales)
            if scales is not None
            else np.zeros((input_shape[1],)),
        }
        return (input_shape[0], input_shape[0]), params

    @t_wrapper
    def apply_fun(params, x1, x2, remat=False):
        """
        :param remat: if True, slam the squared distance calculaation with a remat to 
        prevent XLA fusion bug w/ x64.
        """
        sd = _squared_distance if not remat else _remat_squared_distance
        return params["variance"] * np.exp(-sd(x1, x2, scales=params["scales"]))

    @t_wrapper
    def safe_apply_func(params, x1, x2):
        # "Safe" version that doesn't cause https://github.com/google/jax/issues/3122
        return vmap(apply_single_fun)(params, x1, x2)

    @t_wrapper
    def apply_diag_fun(params, x):
        return params["variance"] * np.ones(x.shape[0])

    @t_wrapper
    def apply_single_fun(params, x1, x2):
        """
        Maps a pair of 1D vectors to a scalar (use this for grads)
        """
        dr = (x1 - x2) / params["scales"]
        r2 = np.dot(dr, dr)
        return params["variance"] * np.exp(-r2)

    return {
        "init": init_fun,
        "apply": apply_fun,
        "apply_diag": apply_diag_fun,
        "apply_single": apply_single_fun,
        "safe_apply": safe_apply_func,
    }
