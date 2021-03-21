# File: distributions.py
# Created Date: 2021-03-14
# Author: Steven Atkinson (steven@atkinson.mn)

import jax.numpy as np

from .functions import logdetchol, trtrs


def mvnorm_logpdf(x, loc, scale_tril):
    const = -0.5 * loc.size * np.log(2.0 * np.pi) - 0.5 * logdetchol(scale_tril)
    a = trtrs(scale_tril, (loc - x)[:, np.newaxis])
    return const - 0.5 * np.sum((a * a))
