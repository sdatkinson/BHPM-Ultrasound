# File: ultrasound.py
# Created Date: 2020-06-25
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Utilities for the ultrasound experiment
"""

from collections import namedtuple
import math
import os

from jax import jit
import jax.numpy as jnp
from jax.random import PRNGKey, split, permutation
from scipy.io import loadmat

Data = namedtuple("Data", ("x", "y"))


class _Architecture(object):
    def __init__(self):
        self.leaf_type = "nn"
        self.root_type = "gp"


architecture = _Architecture()

f_grads = (
    (0, (0,)),  # u_x
    (0, (1,)),  # u_y
    (0, (2,)),  # u_t
    (0, (0, 0)),  # u_xx
    (0, (1, 1)),  # u_yy
    (0, (0, 1)),  # u_xy
)
target_grad = (0, (2, 2))  # u_tt

_us_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ultrasound_data")

_info = {
    "pristine": {
        "filename": os.path.join(_us_dir, "Hsample SAW 5MHz n2", "wvf.mat",),
        "crop": ((200, 350), (100, 150), (325, None)),  # T,Y,X
    },
    "cracked": {
        "filename": os.path.join(
            _us_dir,
            "30Jan15 Nist crack1 240x240 12x12mm avg20 5MHz 0deg grips",
            "wvf.mat",
        ),
        "crop": ((500, 750), (None, None), (None, None)),
    },
}


def get_data(data_type):
    """
    Data are unrolled from a (NT,NY,NX) cube (before subsampling) such that a slice like
    data.reshape((nt, ny, nx))[i] can be plt.imshow()'d

    :return: tuple of (xyt, u) pairs
    """
    info = _info[data_type]
    filename = info["filename"]
    wvf_mat = jnp.array(loadmat(filename)["wvf"])  # Stored as (NY,NX,NT)
    wvf = jnp.transpose(wvf_mat, (2, 0, 1))  # (NT,NY,NX)
    # crop:
    crop = info["crop"]
    wvf = wvf[crop[0][0] : crop[0][1], crop[1][0] : crop[1][1], crop[2][0] : crop[2][1]]
    nt, ny, nx = wvf.shape
    print("Loaded %i measurements after cropping" % wvf.size)

    # Unroll
    xy_scale = 0.05  # mm
    t_scale = 0.02  # usec
    x = jnp.tile(xy_scale * jnp.arange(nx)[None, None, :], (nt, ny, 1))
    y = jnp.tile(xy_scale * jnp.arange(ny - 1, -1, -1)[None, :, None], (nt, 1, nx))
    t = jnp.tile(t_scale * jnp.arange(nt)[:, None, None], (1, ny, nx))

    xyt = jnp.stack((x.flatten(), y.flatten(), t.flatten())).T
    u = wvf.reshape((-1, 1))

    return (Data(xyt, u),)


class DataLoader(object):
    """
    Efficiently feed shuffled minibatches for leaves & root without end.

    Epoch is ill-defined since root & leaf have different batch sizes.
    """

    def __init__(self, datasets, batch_size, shuffle=True, rng=None):
        self.datasets = datasets
        self.shuffle = shuffle
        self._rng = PRNGKey(41) if rng is None else rng
        self._batch_size = batch_size
        n_min = min([len(d.x) for d in self.datasets])
        if n_min < batch_size:
            raise ValueError(
                "Batch size is %i, but the smallest dataset is %i" % (batch_size, n_min)
            )
        else:
            self._batches_per_epoch = n_min // batch_size

    def __iter__(self):
        def shuffle_single(rng, d):
            rng, rng_perm = split(rng)
            i = permutation(rng_perm, len(d.x))
            return Data(d.x[i], d.y[i])

        @jit
        def shuffle(rng, data):
            rng, rng_s = split(rng)
            return tuple(
                [
                    shuffle_single(r, d)
                    for r, d in zip(split(rng_s, num=len(data)), data)
                ]
            )

        def generator(rng, datasets):
            while True:
                rng, rng_shuffle = split(rng)
                if self.shuffle:
                    datasets = shuffle(rng_shuffle, datasets)
                for b in range(self._batches_per_epoch):
                    i = b * self._batch_size
                    j = i + self._batch_size  # Doesn't use last partial batch
                    yield tuple([Data(d.x[i:j], d.y[i:j]) for d in datasets])

        self._rng, rng = split(self._rng)
        return generator(rng, self.datasets)
