# File: transforms.py
# Created Date: 2020-04-19
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Helper for transforming parameters
"""

from functools import wraps
from typing import Iterable

import jax.numpy as np


def apply_transform(params, t):
    """
    Recurse through a dict of parameters, transforming any whose names start 
    with "raw"
    """

    def _pass(params, t):
        return params

    def _apply_transform_dict(params, t):
        transformed = {}
        for key, val in params.items():
            if key.startswith("raw_"):
                assert isinstance(val, np.ndarray)
                key, val = key[4:], t(val)
            else:
                val = apply_transform(val, t)
            transformed[key] = val
        return transformed

    def _apply_transform_list(params, t):
        return [apply_transform(p, t) for p in params]

    def _apply_transform_tuple(params, t):
        return tuple(_apply_transform_list(params, t))

    if not callable(t):
        raise ValueError("Provided transform t must be callable")

    # TODO lists & tuples
    _map = {
        dict: _apply_transform_dict,
        tuple: _apply_transform_tuple,
        list: _apply_transform_list,
        np.ndarray: _pass,
    }
    key = type(params) if not isinstance(params, np.ndarray) else np.ndarray
    if key not in _map:
        msg = "Argument params is invalid type %s" % key
        msg += "\nAllowed types:"
        for k in _map:
            msg += "\n%s" % k
        raise TypeError(msg)
    return _map[key](params, t)


def transform_params(f, transform):
    """
    Decorator for transforming parameters to functions
    """

    @wraps(f)
    def wrapped(params, *args, **kwargs):
        params = apply_transform(params, transform)
        return f(params, *args, **kwargs)

    return wrapped


# Now, implementations of useful tansforms (forward & inverse)


def lower_cholesky(lower=1.0e-14):
    def forward(x):
        return np.diag(np.exp(np.diag(x))) + np.tril(x, -1)

    def inverse(x):
        return np.diag(np.log(np.diag(x) + lower)) + np.tril(x, -1)

    return {"forward": forward, "inverse": inverse}
