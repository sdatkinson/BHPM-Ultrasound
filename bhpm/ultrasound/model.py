# File: model.py
# Created Date: 2020-07-14
# Author: Steven Atkinson (steven@atkinson.mn)

from collections import namedtuple
from functools import partial

from jax import grad, jit, vmap, value_and_grad
from jax.experimental import optimizers, stax
import jax.nn.initializers
import jax.numpy as jnp
from jax.random import PRNGKey, split, permutation
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..jax_code.functions import jitchol
from ..jax_code.models import fcnn
from ..jax_code.models import gp_mean_functions, kernels, likelihoods, gp, sparse_layer
from ..jax_code.models import transforms
from ..jax_code.models.train import cosine_scheduler
from ..jax_code.util import kmeans_centers, batch_apply

from . import Data, DataLoader

NUM_INDUCING = 128
LeafParams = namedtuple("LeafParams", ("u", "a"))

# Describe the structure of the HPM:
_Structure = namedtuple(
    "Structure",
    (
        # List of tuples specifying all u outputs needed for physics,
        # including the non-grad "()"
        "u_operators",
        # Tuple of indices indicating which u ops  will be used for inputs to latent
        # field a() beyond x,y:
        "a_inputs",
        # Which to use for inputs to f():
        "f_inputs",
    ),
)

# Simple wave equation
_structure_1 = _Structure(
    # div
    ("div",),  # ((0, 0), (1, 1)),
    # a(x, y)
    (),
    # f(div)
    (0,),  # (0, 1),
)

# Larger hypothesis space
_structure_2 = _Structure(
    # u,
    ((), "grad_sq", "div"),
    # a(x, y)
    (),
    # f(u, |g|, div)
    (0, 1, 2),
)

_wave_xform = jnp.exp


def wave():
    """
    "Model" API for the wave equation

    params:
        raw likelihood scale
    """

    def init_fun():
        return (-1,), (jnp.array(0.0),)

    def gaussian_fun(params, rng, inputs):
        if inputs.shape[1] != 1:
            raise ValueError("Wave takes in div(u) only")
        outputs = inputs.squeeze()
        return outputs, _wave_xform(params[0]) + jnp.zeros(outputs.shape)

    return {"init": init_fun, "gaussian": gaussian_fun}


def _hpm(structure, datasets, root_type="gp"):
    """
    Custom hidden physics model for the ultrasound problem.

    Leaves have a(x,y) and u(x,y,t).
    Both are MLPs.

    Physics is
    utt = a(x, y) f(...),
    where f is a GP or the wave operator.

    :return: HPM
    """

    n_leaves = len(datasets)
    n = sum([d[0].shape[0] for d in datasets])

    # Public funcs
    u_funcs = fcnn.fcnn(1, layers=5, units=128, first_layer_factor=1)
    a_funcs = fcnn.fcnn(
        1,
        layers=5,
        units=128,
        first_layer_factor=8.0,
        output_activation=stax.serial(
            stax.elementwise(lambda x: 0.1 * x), stax.elementwise(jnp.exp)
        ),
    )
    if root_type == "gp":
        _root_mean = gp_mean_functions.linear()
        _root_kernel = kernels.rbf()
        _root_likelihood = likelihoods.gaussian()
        f_funcs = sparse_layer.sparse_layer(
            n, _root_mean, _root_kernel, _root_likelihood, safe=False, jitter=1.0e-5
        )
    elif root_type == "wave":
        if not (
            len(structure.f_inputs) == 1
            and structure.u_operators[structure.f_inputs[0]] == "div"
        ):
            raise ValueError(
                'Wave operator requires "div" be the only u operator fed to f.'
            )
        f_funcs = wave()

    def init_fun(rng):
        params = {"leaf": [], "root": None}
        for _ in range(n_leaves):
            rng, rng_u, rng_a = split(rng, num=3)
            u_params = u_funcs["init"](rng_u, (-1, 3))[1]  # x, y, t
            a_params = a_funcs["init"](rng_a, (-1, 2 + len(structure.a_inputs)))[
                1
            ]  # x, y, ...
            params["leaf"].append(LeafParams(u_params, a_params))
        rng, rng_root = split(rng)
        if root_type == "gp":
            params["root"] = f_funcs["init"](
                rng_root, (-1, len(structure.f_inputs)), m=NUM_INDUCING
            )[1]
        elif root_type == "wave":
            params["root"] = f_funcs["init"]()[1]
        return params

    def apply_u_ops(params, inputs):
        """
        :param params: For u net
        :param inputs: (N, 3)
        :return: jnp.array, shape (N_Ops,N)
        """
        return jnp.stack([op(params, inputs) for op in u_ops]).T

    def apply_lhs(params, inputs, leaf):
        """
        Compute utt(inputs) for a specified leaf

        :param params: For the whole HPM
        :param inputs: (N,3)
        :return: (N,)
        """
        u_params = params["leaf"][leaf].u
        return utt(u_params, inputs)

    def apply_rhs(params, inputs, leaf, rng=None):
        """
        Compute a^2()f() for a specified leaf.

        Use the mean of f for now.

        :params inputs: (N,3)
        :return: (N,)
        """
        rng = PRNGKey(42) if rng is None else rng
        f_params = params["root"]
        u_params, a_params = params["leaf"][leaf]
        u_op_vals = apply_u_ops(u_params, inputs)
        f_inputs = u_op_vals[:, structure.f_inputs]
        a_inputs = jnp.concatenate(
            (inputs[:, :2], u_op_vals[:, structure.a_inputs]), axis=1
        )
        a = a_funcs["apply"](a_params, None, a_inputs)[:, 0]  # (N,)
        rng, rng_f = split(rng)
        # Posterior mean, flattened from (N,1) to (N,)
        f = f_funcs["gaussian"](f_params, rng_f, f_inputs)[0].squeeze()
        return a * a * f

    def train(
        rng, params, datasets, u_iters=None, af_iters=None, freeze_f=False,
    ):
        """
        Train u's to data
        Train a and f to physics
        """

        rng, rng_leaf = split(rng)
        params["leaf"] = tuple(
            [
                LeafParams(_train_u(rng_u, pl.u, data, iters=u_iters), pl.a,)
                for rng_u, pl, data in zip(
                    split(rng_leaf, num=n_leaves), params["leaf"], datasets
                )
            ]
        )

        # Root pre-training: freeze u's.
        rng, rng_af = split(rng)
        a_params, f_params = _train_af(
            params, rng_af, datasets, iters=af_iters, freeze_f=freeze_f,
        )
        params["leaf"] = tuple(
            [LeafParams(pl.u, ap) for pl, ap in zip(params["leaf"], a_params)]
        )
        params["root"] = f_params

        return params

    def loss_fun(
        params,
        batches,
        rng,
        leaf_ns,
        root_batch=None,
        loss_type="nlp",
        u_loss=True,
        af_loss=True,
    ):
        """
        * Data: u targets
        * Physics: utt = a^2 * f

        :param leaf_ns: (tuple of ints) How many data in each leaf.
        :param root_batch: n per leaf to use in root
        """
        # Leaves:
        leaves_loss = 0.0
        a_list, f_in_list, lhs_list = [], [], []
        for lp, batch, leaf_n in zip(params["leaf"], batches, leaf_ns):
            # Data loss:
            rng, rng_u = split(rng)
            leaves_loss = leaves_loss + u_funcs["loss"](
                lp.u,
                rng_u,
                batch,
                n=leaf_n,
                loss_type=loss_type,
                reduce="sum",  # mean later
            )
            # Gather inputs needed for root/physics
            xt_root = batch.x[:root_batch]
            u_op_vals = apply_u_ops(lp.u, xt_root)
            f_in_list.append(u_op_vals[:, structure.f_inputs])
            # columns are 0,3,4 = u, uxx, uyy
            a_in = jnp.concatenate(
                (xt_root[:, :2], u_op_vals[:, structure.a_inputs]), axis=1
            )
            a_list.append(a_funcs["apply"](lp.a, None, a_in)[:, 0])  # (B_root,)
            lhs_list.append(utt(lp.u, xt_root))

        # Physics loss
        f_in = jnp.concatenate(f_in_list)
        rng, rng_f = split(rng)
        f_mean, f_std = f_funcs["gaussian"](params["root"], rng_f, f_in)
        f_mean, f_std = f_mean.squeeze(), f_std.squeeze()  # NN returns (N,1)'s...
        a = jnp.concatenate(a_list)
        rhs_mean, rhs_std = a * a * f_mean, a * a * f_std
        lhs = jnp.concatenate(lhs_list)

        # Yuck :( TODO into own func w/ root_type, loss_type to dispatch...
        if loss_type == "mse":
            root_loss = jnp.power(lhs - rhs_mean, 2).mean() * n
        else:  # nlp
            if root_type == "gp":  # GP
                lik_scale = transforms.apply_transform(
                    params["root"], sparse_layer.param_transform
                )["likelihood"]["noise"]
                root_loss = -(
                    n * norm.logpdf(lhs, loc=rhs_mean, scale=f_std + lik_scale).mean()
                    - f_funcs["kl"](params["root"])
                )
            elif root_type == "wave":
                lik_scale = _wave_xform(params["root"][0])
                root_loss = -(
                    n * norm.logpdf(lhs, loc=rhs_mean, scale=f_std + lik_scale).mean()
                )
        # Also yuck; can shovel into a loss aggregation func.
        if u_loss and af_loss:
            combined_loss = leaves_loss + root_loss
        elif u_loss:
            combined_loss = leaves_loss
        else:
            combined_loss = root_loss
        loss = combined_loss / n
        return loss

    def print_mses(params, datasets, rng=None):
        """
        Quick helper to show about how good the data & physics are learned.
        Current just takes a single minibatch instead of the actual full datasets.
        """
        rng = PRNGKey(42) if rng is None else rng
        leaf_ns = [len(d.x) for d in datasets]
        kwargs = {"loss_type": "mse"}
        for batch in DataLoader(datasets, 16384):
            args = (params, batch, rng, leaf_ns)
            print("u  MSE : %.2e" % loss_fun(*args, af_loss=False, **kwargs))
            print("af MSE : %.2e" % loss_fun(*args, u_loss=False, **kwargs))
            break

    # Private funcs
    def _train_u(
        rng,
        params,
        data,
        batch_size=4096,
        iters=None,
        plot_every=1000,
        checkpoint_every=1000,
    ):
        """
        Train observation function on data, using cross-validation to early stop
        """
        # Could definitely just pull this out as a general-purpose NN-training func and
        # give things space to breathe...
        # Collapse in your editor is your friend!

        def train_test_split(data, rng=None, n_test=None):
            """
            Create a train-test split
            """
            rng = PRNGKey(42) if rng is None else rng
            n = len(data.x)
            rng, rng_perm = split(rng)
            i = permutation(rng, n)
            n_test = min(16384, int(0.1 * n)) if n_test is None else n_test
            if isinstance(n_test, float):
                n_test = int(n_test * n)
            n_train = n - n_test
            i_train, i_test = i[:n_train], i[n_train:]
            return (
                Data(data.x[i_train], data.y[i_train]),
                Data(data.x[i_test], data.y[i_test]),
            )

        show = plot_every > 0
        iters = 20000 if iters is None else iters
        data_train, data_test = train_test_split(data)
        n_train = len(data_train.x)
        dataloader = DataLoader((data_train,), batch_size)
        opt_init, opt_update, opt_params = optimizers.adam(
            cosine_scheduler(0.001, 0.0003, iters)
        )
        state = opt_init(params)

        @jit
        def fstep(i, state, batch):
            f, g = value_and_grad(u_funcs["loss"])(
                opt_params(state),
                None,
                batch,
                n=n_train,
                loss_type="nlp",
                reduce="mean",
            )
            return f, opt_update(i, g, state)

        jitloss = jit(partial(u_funcs["loss"], reduce="mean"))

        Checkpoint = namedtuple("Checkpoint", ("iter", "score", "params"))
        checkpoints = []
        if show:
            losses = []
            fig = plt.figure()
            ax = fig.gca()
            lines = None
        for i, batch in tqdm(enumerate(dataloader, 1), total=iters):
            if i > iters:
                break
            f, state = fstep(i, state, batch[0])
            if i % checkpoint_every == 0:
                rng, rng_loss = split(rng)
                checkpoints.append(
                    Checkpoint(
                        i,
                        jitloss(opt_params(state), rng_loss, data_test),
                        opt_params(state),
                    )
                )
            if show:
                losses.append(f)
                if i % plot_every == 0:
                    x_train_line, y_train_line = (
                        jnp.arange(1, len(losses), 100),
                        jnp.array(losses[::100]),
                    )
                    x_test_line, y_test_line, _ = zip(*checkpoints)
                    if lines is None:
                        lines = (
                            plt.plot(
                                x_train_line, y_train_line, linestyle="none", marker="."
                            )[0],
                            plt.plot(x_test_line, y_test_line)[0],
                        )
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Loss")
                    else:
                        lines[0].set_data(x_train_line, y_train_line)
                        lines[1].set_data(x_test_line, y_test_line)
                    ax.set_xlim(
                        x_train_line.min(), max(max(x_train_line), max(x_test_line))
                    )
                    y_min = min(min(y_train_line), min(y_test_line))
                    ax.set_yscale("symlog" if y_min < 0.0 else "log")
                    ax.set_ylim(y_min, y_train_line[:10].max())
                    plt.pause(0.001)

        return (
            checkpoints[np.argmin([c.score for c in checkpoints])].params
            if len(checkpoints) > 0
            else opt_params(state)
        )

    def _reinit_gp_root(params, datasets):
        """
        Helper to reinitialize a GP root with better initial guesses.
        Assumes rbf kernel

        :param params: For whole BHPM
        :param datasets: all datasets.
        :return: The BHPM params
        """
        # Avoid computing all x's & y's here if possible.
        x = jnp.concatenate(
            [
                batch_apply(
                    partial(
                        lambda params, inputs: apply_u_ops(params, inputs)[
                            :, structure.f_inputs
                        ],
                        lp.u,
                    ),
                    data.x,
                    32000,
                )
                for lp, data in zip(params["leaf"], datasets)
            ]
        )

        # Mean function...
        # Kernel...
        params["root"]["kernel"]["raw_scales"] = jnp.log(x.max(axis=0) - x.min(axis=0))
        # Inducings
        xu = kmeans_centers(x, params["root"]["xu"].shape[0])
        params["root"]["xu"] = xu
        params["root"]["q_mu"] = jnp.zeros((len(xu),))
        params["root"]["raw_q_sqrt"] = transforms.lower_cholesky()["inverse"](
            0.1 * jitchol(_root_kernel["apply"](params["root"]["kernel"], xu, xu))
        )
        # Likelihood...

        return params

    def _train_af(
        params,
        rng,
        datasets,
        iters=None,
        leaf_batch=8192,
        root_batch=8192,
        plot_every=1000,
        freeze_f=False,
    ):
        """
        Train a and f while keeping u frozen.
        """
        iters = 20000 if iters is None else iters
        show = plot_every is not None
        leaf_ns = tuple([len(d.x) for d in datasets])
        root_batch_per = root_batch // len(datasets)
        dataloader = DataLoader(datasets, leaf_batch)
        if root_type == "gp" and not freeze_f:
            params = _reinit_gp_root(params, datasets)
        # Repack params to "freeze" u...
        u_params = tuple([pl.u for pl in params["leaf"]])
        if freeze_f:
            af_params = tuple([pl.a for pl in params["leaf"]])
        else:
            af_params = (tuple([pl.a for pl in params["leaf"]]), params["root"])

        opt_init, opt_update, opt_params = optimizers.adam(
            cosine_scheduler(0.001, 0.0003, iters)
        )

        state = opt_init(af_params)

        def loss_af(af_params, batch, rng):
            if freeze_f:
                a_params = af_params
                f_params = params["root"]
            else:
                a_params, f_params = af_params
            p = {
                "leaf": tuple(
                    [LeafParams(up, ap) for up, ap in zip(u_params, a_params)]
                ),
                "root": f_params,
            }
            return loss_fun(
                p,
                batch,
                rng,
                leaf_ns,
                root_batch=root_batch_per,
                loss_type="nlp",
                u_loss=False,
            )

        @jit
        def fstep(i, state, batch, rng):
            f, g = value_and_grad(loss_af)(opt_params(state), batch, rng)
            return f, opt_update(i, g, state)

        if show:
            fig = plt.figure()
            ax = fig.gca()
            line = None
            losses = []
        for i, batch in enumerate(tqdm(dataloader, total=iters), 1):
            if i > iters:
                break
            rng, rng_step = split(rng)
            loss, state = fstep(i, state, batch, rng_step)
            if show:
                losses.append(loss)
                if i % plot_every == 0:
                    x_line, y_line = (
                        jnp.arange(1, len(losses), 100),
                        jnp.array(losses[::100]),
                    )
                    if line is None:
                        line = plt.plot(x_line, y_line, linestyle="none", marker=".")[0]
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Loss")
                    else:
                        line.set_data(x_line, y_line)
                    ax.set_xlim(x_line.min(), x_line.max())
                    y_min = y_line.min()
                    ax.set_yscale("symlog" if y_min < 0.0 else "log")
                    ax.set_ylim(y_min, y_line[:10].max())
                    plt.pause(0.001)

        if freeze_f:
            return opt_params(state), params["root"]
        else:
            return opt_params(state)

    def _grad_op(f, grads):
        """
        Apply grads (partial derivatives actually) to an op

        :return: apply(params, inputs) -> (N,)
        """
        f_single = fcnn.as_single(f)
        for g in grads:
            f_single = fcnn.ddx(f_single, g)
        g_vmap = vmap(f_single, in_axes=(None, None, 0))

        def g(params, inputs):
            return g_vmap(params, None, inputs).squeeze()

        return g

    def _div(f):
        """
        Construct divergence operator
        """
        fxx = _grad_op(f, (0, 0))
        fyy = _grad_op(f, (1, 1))

        def g(params, inputs):
            return fxx(params, inputs) + fyy(params, inputs)

        return g

    def _grad_sq(f):
        """
        Construct gradient's squared magnitude operator
        """
        fx = _grad_op(f, (0,))
        fy = _grad_op(f, (1,))

        def g(params, inputs):
            return jnp.power(fx(params, inputs), 2) + jnp.power(fy(params, inputs), 2)

        return g

    def _construct_op(f, op):
        """
        Construct either a partial derivative or a compositional operator (div or grad squared)
        """
        if isinstance(op, tuple):
            return _grad_op(f, op)
        elif op == "grad_sq":
            return _grad_sq(f)
        elif op == "div":
            return _div(f)
        else:
            return ValueError("Operator %s not recognized" % str(op))

    # Private data:
    u_ops = tuple([_construct_op(u_funcs["apply"], op) for op in structure.u_operators])
    utt = _grad_op(u_funcs["apply"], (2, 2))  # utt

    return {
        "u_funcs": u_funcs,
        "a_funcs": a_funcs,
        "f_funcs": f_funcs,
        "init": init_fun,
        "apply_u_ops": apply_u_ops,
        "apply_lhs": apply_lhs,
        "apply_rhs": apply_rhs,
        "train": train,
        "loss": loss_fun,
        "print_mses": print_mses,
    }


hpm_1 = partial(_hpm, _structure_1)
hpm_2 = partial(_hpm, _structure_2)
