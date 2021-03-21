# File: likelihoods.py
# Created Date: 2021-03-12
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Likelihoods
"""

from functools import partial

import jax.numpy as np

from .transforms import transform_params


def gaussian():
    t_wrapper = partial(transform_params, transform=np.exp)

    def init_fun(rng, input_shape, noise=None):
        return (
            input_shape,
            {"raw_noise": np.log(noise) if noise is not None else np.array(-2.0)},
        )

    @t_wrapper
    def apply_fun(params, mean, cov):
        if cov.ndim == 1:
            cov = cov + params["noise"]
        else:
            cov = cov + params["noise"] * np.eye(*cov.shape)
        return mean, cov

    return {"init": init_fun, "apply": apply_fun}
