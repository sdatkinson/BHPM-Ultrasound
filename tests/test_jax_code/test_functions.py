# File: test_functions.py
# Created Date: 2019-11-17
# Author: Steven Atkinson (steven@atkinson.mn)

import os
import sys

import jax.numpy as np
import pytest

from bhpm.jax_code import functions


def test_logdetchol():
    a = np.eye(3)
    val = functions.logdetchol(a)
    assert val == 0.0  # Because ID mtx


def test_cholesky():
    a = np.eye(3)
    functions.cholesky(a)


def test_trtrs():
    a, b = np.eye(3), np.ones((3, 2))
    x = functions.trtrs(a, b)

    assert x.shape == b.shape
