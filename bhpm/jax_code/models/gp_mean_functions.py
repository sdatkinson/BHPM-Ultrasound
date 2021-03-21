# File: gp_mean_functions.py
# Created Date: 2020-04-20
# Author: Steven Atkinson (steven@atkinson.mn)

from jax.experimental import stax
import jax.numpy as np


def zero():
    def init_fun(rng, input_shape):
        return (input_shape[0],), ()

    def apply_fun(params, inputs):
        return np.zeros((inputs.shape[0],))

    return {"init": init_fun, "apply": apply_fun}


def constant():
    def init_fun(rng, input_shape):
        return (input_shape[0],), (np.zeros((1,)))

    def apply_fun(params, inputs):
        return params[0] + np.zeros((inputs.shape[0],))

    return {"init": init_fun, "apply": apply_fun}


def linear():
    """
    Thin wrapper arund stax linear layer so that outputs are 1D (to match w/ 
    GPs) and return a dict of functions
    """
    init_fun, _apply_fun = stax.Dense(1)

    def apply_fun(params, inputs):
        return _apply_fun(params, inputs).squeeze()

    return {"init": init_fun, "apply": apply_fun}
