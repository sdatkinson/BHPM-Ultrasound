# File: fcnn.py
# Created Date: 2020-04-20
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Fully-connected network stuff
"""

from functools import partial
import math

from jax import grad, random
from jax.experimental import stax
import jax.nn.initializers
import jax.numpy as np
from jax.scipy.stats import norm

from .transforms import transform_params

positive_transform = np.exp
t_wrapper = partial(transform_params, transform=positive_transform)


def siren_w_init(factor=None):
    """
    Initialization from Sitzmann et al.,
    "Implicit Neural Representations with Periodic Activation Functions"
    (2020)
    for MLPs with sine activations.

    When used on the first layer, `factor` can be thought of as a sort of (inverse)
    length scale of the MLP. Sitzmann recommends factor=30 for first layer, but this can
    cause severe overfitting when you have sparse data and regularization via a smaller
    factor can encourage better regularization/generalization.
    """
    factor = 1.0 if factor is None else factor
    # scale=2.0 causes weights to init from U[-sqrt(6/n), sqrt(6/n)]
    # 1800 = multiply by 30^2 as I understand the paper suggests
    # (square since scale is inside the sqrt)
    return partial(
        jax.nn.initializers.variance_scaling, 2.0 * factor ** 2, "fan_in", "uniform",
    )


Dense = stax.Dense


def skip_connect(layer_funcs):
    init_fun, _apply_fun = layer_funcs

    def apply_fun(params, inputs, **kwargs):
        return inputs + _apply_fun(params, inputs, **kwargs)

    return init_fun, apply_fun


def reduce_add(*inputs):
    output = 0.0
    for x in inputs:
        output = output + x
    return output


def parallel(*layers, reduce_fun=reduce_add):
    """
    Execute layers in parallel then reduce them back together
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        rngs = random.split(rng, num=nlayers)
        params, out_shape = [], None
        for r, layer_init in zip(rngs, init_funs):
            o, p = layer_init(r, input_shape)
            params.append(p)
            out_shape = o if out_shape is None else out_shape
            if o != out_shape:
                raise ValueError("Layers don't have same output shapes; can't reduce")
        return out_shape, params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        outputs = [
            f(p, inputs, rng=r, **kwargs) for f, p, r in zip(apply_funs, params, rngs)
        ]
        return reduce_fun(*outputs)

    return init_fun, apply_fun


def fcnn(
    output_dim: int,
    layers=2,
    units=256,
    skips=False,
    output_activation=None,
    first_layer_factor=None,
    parallel_with=None,
):
    """
    Creates init and apply functions for a fully-connected NN with a Gaussian
    likelihood.

    :param output_dim: Number of output dimensions
    :param first_layer_factor: Scaling factor for first dense layer initialization.
    """

    first_layer_factor = 10.0 if first_layer_factor is None else first_layer_factor

    activation = stax.elementwise(np.sin)
    block = (
        skip_connect(stax.serial(Dense(units, W_init=siren_w_init()()), activation))
        if skips
        else stax.serial(Dense(units, W_init=siren_w_init()()), activation)
    )

    layer_list = [
        stax.serial(
            Dense(
                units,
                W_init=siren_w_init(factor=first_layer_factor)(),
                b_init=jax.nn.initializers.normal(stddev=2.0 * math.pi),
            ),
            activation,
        ),
        *([block] * (layers - 1)),
        Dense(output_dim),  # No skips on the last one!
    ]
    if output_activation is not None:
        layer_list.append(output_activation)
    if parallel_with is None:
        _init_fun, _apply_fun = stax.serial(*layer_list)
    else:
        _init_fun, _apply_fun = parallel(stax.serial(*layer_list), parallel_with)

    def init_fun(rng, input_shape):
        output_shape, net_params = _init_fun(rng, input_shape)
        params = {"net": net_params, "raw_noise": np.array(-2.0)}
        return output_shape, params

    # Conform to API:
    @t_wrapper
    def apply_fun(params, rng, inputs):
        return _apply_fun(params["net"], inputs)

    @t_wrapper
    def gaussian_fun(params, rng, inputs):
        pred_mean = apply_fun(params, None, inputs)
        pred_std = params["noise"] * np.ones(pred_mean.shape)
        return pred_mean, pred_std

    @t_wrapper
    def loss_fun(
        params, rng, data, batch_size=None, n=None, loss_type="nlp", reduce="sum"
    ):
        """
        :param batch_size: How large a batch to subselect from the provided data
        :param n: The total size of the dataset (to multiply batch estimate by)
        """
        assert loss_type in ("nlp", "mse")
        inputs, targets = data
        n = inputs.shape[0] if n is None else n
        if batch_size is not None:
            rng, rng_batch = random.split(rng)
            i = random.permutation(rng_batch, n)[:batch_size]
            inputs, targets = inputs[i], targets[i]
        preds = apply_fun(params, rng, inputs).squeeze()
        mean_loss = (
            -norm.logpdf(targets.squeeze(), preds, params["noise"]).mean()
            if loss_type == "nlp"
            else np.power(targets.squeeze() - preds, 2).mean()
        )
        if reduce == "sum":
            loss = n * mean_loss
        elif reduce == "mean":
            loss = mean_loss
        return loss

    def sample_fun_fun(rng, params):
        def f(x):
            return apply_fun(params, rng, x)

        return f

    return {
        "init": init_fun,
        "apply": apply_fun,
        "gaussian": gaussian_fun,
        "loss": loss_fun,
        "sample_function": sample_fun_fun,
    }


def ddx(f, dim):
    """
    Input: a scalarized function of the form y=f(params, x)
    Output: scalarized function df(param, input) computing df/dx_dim
    """
    return lambda params, rng, x: grad(f, argnums=2)(params, rng, x)[dim]


def as_single(apply_fun):
    """
    Turn (N,DX) -> (N,1) fun into a (DX,) -> () so that we can take grads and
    vmap later.
    """
    return lambda params, rng, x: apply_fun(params, rng, x[None])[0, 0]
