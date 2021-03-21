# File: train.py
# Created Date: 2020-05-12
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Training utilities
"""

import math

import jax.numpy as np


def cosine_scheduler(lr_start, lr_end, iters):
    if iters == 1:
        step_size = lr_start
    else:

        def step_size(i):
            amp = 0.5 * (lr_start - lr_end)
            omega = math.pi / (iters - 1)
            return lr_end + amp * (1.0 + np.cos(omega * i))

    return step_size
