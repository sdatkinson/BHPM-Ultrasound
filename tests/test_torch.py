# File: test_torch.py
# Created Date: 2021-03-14
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest

try:
    import torch

    have_torch = True
except:
    have_torch = False


def _cuda():
    torch.nn.Linear(128, 2).to("cuda")(torch.randn(16, 128).to("cuda"))


if have_torch and torch.cuda.is_available():
    test_cuda = _cuda
else:
    test_cuda = pytest.mark.xfail(_cuda)
