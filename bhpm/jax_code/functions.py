# File: functions.py
# Created Date: 2019-11-16
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Helpful functions
"""

from jax import lax
import jax.numpy as np
from jax.scipy import linalg


def logdetchol(a):
    """
    Log-determinant based on the cholesky of the matrix of interest
    """

    return 2.0 * np.sum(np.log(np.diag(a)))


def cholesky(a, lower=True, **kwargs):
    return linalg.cholesky(a, lower=lower, **kwargs)


def jitchol(a):
    """
    Cholesky that adds jitter on failure.
    """

    def _safe_cholesky(a):
        c = cholesky(a)
        if np.any(np.isnan(c)):
            raise RuntimeError()
        return c

    try:
        return _safe_cholesky(a)
    except RuntimeError:
        pass
    j = np.mean(np.diag(a))
    for i in range(10):
        try:
            return _safe_cholesky(a + j * np.power(0.1, 10 - i) * np.eye(*a.shape))
        except RuntimeError:
            pass
    raise RuntimeError("Cholesky failed!")


def jitchol2(x, jitter=1.0e-6):
    """
    A stopgap that tries to add jitter once.
    This can be JIT-ed thanks to jax.lax.cond, but might not be as robust as
    jitchol that tries multiple times before giving up.
    """

    def _jitchol2_2d(x, jitter):
        # Scale jitter to the matrix at hand
        jitter = jitter * np.mean(np.diag(x))
        # return _jc((x, jitter))
        lx = cholesky(x)
        return lax.cond(
            np.any(np.isnan(lx)),
            x + jitter * np.eye(*x.shape),
            cholesky,
            x,
            lambda _x: _x,
        )

    def _jitchol2_3d(x, jitter):
        """
        :param x: (B,N,N)
        """
        # Scale jitter to the matrices at hand
        # (B,1,1)
        jitter = (
            1.0e-12
            + jitter * np.array([np.mean(np.diag(xi)) for xi in x])[:, None, None]
        )
        # return _jc((x, jitter))
        lx = cholesky(x)
        return lax.cond(
            np.any(np.isnan(lx)),
            x + jitter * np.eye(x.shape[1])[None],
            cholesky,
            x,
            lambda _x: _x,
        )

    return {2: _jitchol2_2d, 3: _jitchol2_3d}[x.ndim](x, jitter)


def trtrs(a, b, lower=True):
    # Verify dimensions:
    if a.ndim == 2 and b.ndim == 3:
        # Hacky.
        a = np.tile(a[None], (b.shape[0], 1, 1))
    return linalg.solve_triangular(a, b, lower=lower)


def _test():
    from jax import jit

    x = np.ones((5, 5))
    jitchol(x)
    jitchol2(x)
    jit(jitchol2)(x)


if __name__ == "__main__":
    _test()
