# File: util.py
# Created Date: 2020-05-17
# Author: Steven Atkinson (steven@atkinson.mn)

import jax
import jax.numpy as np
import numpy as onp
from scipy.cluster.vq import kmeans2
from tqdm import tqdm


@jax.custom_transforms
def clip_up(x):
    """
    Special clipping function that clips x up to 0.0

    Used for squared distances because we know that they should never be 
    negative. However, the numerics don't always work out and we want to fix 
    that.

    The problem with using np.clip() is that if it modifies any value, the 
    gradient of that value turns into zero.  If we want to take a second 
    derivative, then that will be zero too, *even though the second derivative
    of the squared-distance operation is a constant positive!  This is a 
    substantial problem in practice and needs a workaround.

    In PyTorch, you could do this using detach, e.g.
    >>> y = x - (torch.clamp(x, max=0.0)).detach()
    But we need something for JAX.
    """

    return np.clip(x, 0.0, None)


jax.defjvp(clip_up, lambda g, ans, x: g)
jax.defvjp(clip_up, lambda g, ans, x: g)


class Dataloader(object):
    def __init__(self, *tensors, batch_size=None):
        self.tensors = tensors
        self.batch_size = self.num_data if batch_size is None else batch_size
        self._shuffled_tensors = None
        self._batches_served = None

    @property
    def num_data(self):
        return len(self.tensors[0])

    @property
    def num_tensors(self):
        return len(self.tensors)

    def shuffle(self, rng):
        # TODO make this not such a hack.  Apparently jax.random.shuffle(r, x)
        # Shuffles every vector differently!!!
        # isn't like x[randperm(n)], which keeps slices together
        order = jax.random.shuffle(rng, np.arange(self.num_data))
        self._shuffled_tensors = tuple([t[order] for t in self.tensors])

    def __iter__(self):
        self._batches_served = 0
        return self

    def __next__(self):
        i_start = self._batches_served * self.batch_size
        i_end = min(i_start + self.batch_size, self.num_data)
        self._batches_served += 1
        if i_start < self.num_data:
            tensors = (
                self._shuffled_tensors
                if self._shuffled_tensors is not None
                else self.tensors
            )
            return tuple([t[i_start:i_end] for t in tensors])
        else:
            raise StopIteration


class DummyArray(object):
    """
    Use with jax.eval_shape; see 
    https://jax.readthedocs.io/en/latest/jax.html#jax.eval_shape
    """

    def __init__(self, shape, dtype=np.float32):
        self.shape, self.dtype = shape, dtype


def rmse(x, y):
    return np.sqrt(np.mean(np.power(x - y, 2)))


def kmeans_centers(x: np.ndarray, k: int, perturb_if_fail: bool = False) -> np.ndarray:
    """
    Use k-means clustering and find the centers of the clusters.
    
    :param x: The data, (N,D)
    :param k: Number of clusters
    :param perturb_if_fail: Move the points randomly in case of a numpy 
        LinAlgError.

    :return: the centers, (K,D)
    """
    try:
        return kmeans2(x, k)[0]
    except onp.linalg.LinAlgError:
        x_scale = x.std(axis=0)
        x_perturbed = x + 1.0e-4 * x_scale * onp.random.randn(*x.shape)
        return kmeans2(x_perturbed, k)[0]


def batch_apply(f, inputs, batch_size, verbose=False):
    """
    Apply a function f on lots of inputs in batches so that we dont' OOM

    :param f: Function f(inputs)
    :param inputs: (N,DIn)
    :param batch_size: (Int)
    :return: outputs via jnp.concatenate
    """
    jf = jax.jit(f)
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))
    it = tqdm(range(num_batches)) if verbose else range(num_batches)
    return np.concatenate(
        [jf(inputs[b * batch_size : (b + 1) * batch_size]) for b in it]
    )
