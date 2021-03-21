# File: post_process.py
# Created Date: 2020-07-15
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Plot figures based on train.py
"""

from argparse import ArgumentParser
from collections import namedtuple
from functools import partial
import json
import os
import pickle

import jax.config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bhpm import ultrasound
from bhpm.jax_code.util import batch_apply
from bhpm.util import plot_triple as _plot_triple, valid_timestamp

# from bhpm.turbo import turbo

LEAF = 0  # We're only doing a single leaf in this...
VMIN_OBS, VMAX_OBS = -10.0, 10.0
VMIN_PHYS, VMAX_PHYS = -1e4, 1e4


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Which run to post-process (default is most recent)",
    )
    return parser.parse_args()


def get_outdir(args):
    base_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.isdir(base_dir):
        raise RuntimeError("No run directory found at %s" % base_dir)
    runs = os.listdir(base_dir)
    if args.date is not None:
        if args.date not in runs:
            raise RuntimeError("Failed to find specified run date %s" % args.date)
        outdir = os.path.join(base_dir, args.date)
    else:
        outdir = os.path.join(
            base_dir,
            sorted([r for r in runs if valid_timestamp(os.path.join(base_dir, r))])[-1],
        )
    return outdir


def get_run_args(outdir):
    filename = os.path.join(outdir, "args.json")
    if not os.path.isfile(filename):
        raise FileNotFoundError("Failed to find run args at %s" % filename)
    with open(filename, "r") as f:
        args_dict = json.load(f)
    keys, vals = zip(*[(k, v) for k, v in args_dict.items()])
    return namedtuple("args", keys)(*vals)


def rmse(x, y):
    return jnp.sqrt(jnp.power(x - y, 2).mean())


def cube_shape(xyt):
    """
    :return: nx, ny, nt
    """
    return [jnp.unique(xyt[:, i]).size for i in range(xyt.shape[1])]


def as_cube(xyt, u):
    """
    :return: (NX,), (NY,), (NT,), (NT, NY, NX)
    """
    nx, ny, nt = cube_shape(xyt)
    u = u.reshape((nt, ny, nx))
    x = xyt[:nx, 2]
    y = xyt[: ny * nx : nx, 1]
    t = xyt[: nt * ny * nx : ny * nx, 0]
    return namedtuple("Cube", ("x", "y", "t", "u"))(x, y, t, u)


def get_params(outdir, hint):
    params_file = os.path.join(outdir, "%s.pkl" % hint)
    if os.path.isfile(params_file):
        with open(params_file, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("Failed to find params at %s" % params_file)


def _plot(z, fname, vmin=-10.0, vmax=10.0, axis="off", extent=None):
    aspect = z.shape[1] / z.shape[0]
    fig = plt.figure(figsize=(2, 2 * (2.0 / 3.0) * (0.2 + 0.8 / aspect)))
    ax = fig.add_subplot(1, 2, 1)
    cax = fig.add_subplot(1, 2, 2)
    s = ax.imshow(z, vmin=vmin, vmax=vmax, extent=extent)  # , cmap=turbo)
    ax.axis(axis)
    plt.colorbar(s, cax=cax)
    ax.set_position((0, 0, 0.79, 1.0))
    cax.set_position((0.8, 0.0, 0.05, 1.0))
    [plt.savefig(fname + ft, bbox_inches="tight") for ft in (".png", ".pdf")]
    plt.close()


def plot_observations(funcs, params, datasets, idx, outdir, leaf=0):
    """
    Data, u net, error

    :return: (float) RMSE
    """
    data = datasets[leaf]
    nx, ny, nt = cube_shape(data.x)
    pred = batch_apply(
        partial(funcs["u_funcs"]["apply"], params["leaf"][leaf].u, None),
        data.x,
        32000,
        verbose=True,
    ).reshape((nt, ny, nx))
    target = data.y.reshape((nt, ny, nx))

    _plot(target[idx], os.path.join(outdir, "data"))
    _plot(pred[idx], os.path.join(outdir, "simulation"))
    _plot(
        jnp.abs(target[idx] - pred[idx]),
        os.path.join(outdir, "obs_error"),
        vmin=0.0,
        vmax=None,
    )
    obs_rmse = rmse(pred, target)
    print("Observation RMSE: %.3e" % obs_rmse)
    return obs_rmse.tolist()


def plot_triple(lhs, rhs, datasets, idx, vmin, vmax, batch_size=32000, savefig=None):
    """
    Plot 3 imshows where first two are "LHS" and "RHS" of something (e.g. observed,
    simulated) and the 3rd panel is the absolute error between the first two

    :param lhs: first panel function
    :param rhs: Second panel function
    """
    data = datasets[LEAF]
    nx, ny, nt = cube_shape(data.x)

    if lhs is None:  # Use observations (hacky)
        z_lhs = data.y.reshape((nt, ny, nx))
    else:
        z_lhs = batch_apply(lhs, data.x, batch_size, verbose=True).reshape((nt, ny, nx))
    z_rhs = batch_apply(rhs, data.x, batch_size, verbose=True).reshape((nt, ny, nx))

    _plot_triple(np.array(z_lhs[idx]), np.array(z_rhs[idx]), vmin, vmax, saveas=savefig)

    lhs_rhs_rmse = rmse(z_lhs, z_rhs)
    print("RMSE: %.3e" % lhs_rhs_rmse)
    return lhs_rhs_rmse.tolist()


def plot_frames(funcs, params, data, outdir, nframes=5, leaf=0):
    """
    Plot a series of 5 frames in time for visualization
    """

    nx, ny, nt = cube_shape(data.x)
    aspect = ny / nx
    figscale = 1.5
    fig = plt.figure(
        figsize=(figscale * nframes + 0.5, 1.3 * aspect * figscale)
    )  # Hack 1.2 for crack

    pred = batch_apply(
        partial(funcs["u_funcs"]["apply"], params["leaf"][leaf].u, None),
        data.x,
        32000,
        verbose=True,
    ).reshape((nt, ny, nx))
    axwidth = 0.9 / nframes

    for i, t_idx in enumerate(jnp.linspace(0, nt - 1, nframes).astype(int)):
        t = 0.02 * t_idx
        ax = fig.add_subplot(1, nframes, i + 1)
        s = ax.imshow(pred[t_idx], vmin=-10.0, vmax=10.0, cmap="bone")
        ax.set_title("$t=%.2f \mu$s" % t)
        ax.axis("off")
        ax.set_position((axwidth * i, 0.0, 0.99 * axwidth, 1.0))
        if i == nframes - 1:
            cax = fig.add_subplot(1, nframes + 1, nframes + 1)
            plt.colorbar(s, cax=cax)
            cax.set_position((nframes * axwidth, 0.1, 0.01, 0.8))
    [
        plt.savefig(os.path.join(outdir, "frames." + figtype))
        for figtype in ("pdf", "png")
    ]


def plot_physics(funcs, params, datasets, idx, outdir, leaf=0):
    """
    Plots for utt(x,y,t) = rhs(x,y,t)
    1. utt()
    2. a^2()f() (right hand side)
    3. Abs difference between the two

    :return: (float) RMSE
    """

    data = datasets[leaf]
    nx, ny, nt = cube_shape(data.x)
    f_utt = partial(funcs["apply_lhs"], params, leaf=leaf)
    f_rhs = partial(funcs["apply_rhs"], params, leaf=leaf)
    utt = batch_apply(f_utt, data.x, 32000, verbose=True).reshape((nt, ny, nx))
    rhs = batch_apply(f_rhs, data.x, 32000, verbose=True).reshape((nt, ny, nx))

    _plot(utt[idx], os.path.join(outdir, "utt"), vmin=-1e4, vmax=1e4)
    _plot(rhs[idx], os.path.join(outdir, "rhs"), vmin=-1e4, vmax=1e4)
    _plot(
        jnp.abs(utt[idx] - rhs[idx]),
        os.path.join(outdir, "phys_error"),
        vmin=0.0,
        vmax=None,
    )
    phys_rmse = rmse(utt, rhs)
    print("Physics RMSE: %.3e" % phys_rmse)
    return phys_rmse.tolist()


def plot_crack(funcs, params, datasets, outdir, leaf=0):
    """
    Have to think about this one since the speed of sound now depends upon what's
    hitting it...
    """
    xyt = datasets[leaf].x
    nx, ny, nt = cube_shape(xyt)
    pred = funcs["a_funcs"]["apply"](
        params["leaf"][leaf].a, None, xyt[: nx * ny, :2]
    ).reshape((ny, nx))
    _plot(pred, os.path.join(outdir, "crack"), vmin=0.0, vmax=None)


def main(args, testing=False):
    main_outdir = get_outdir(args)
    run_args = get_run_args(main_outdir)
    if run_args.model_type == "bhpm":
        print("NOTE: x64 for BHPM (GPs)...")
        jax.config.update("jax_enable_x64", True)

    datasets = ultrasound.get_data(run_args.data_type)
    hpm_type = {"ipinn": ultrasound.hpm_1, "bhpm": ultrasound.hpm_2}[
        run_args.model_type
    ]
    hpm_funcs = hpm_type(
        datasets, root_type={"ipinn": "wave", "bhpm": "gp"}[run_args.model_type]
    )
    for params_fname in [f[:-4] for f in os.listdir(main_outdir) if f.endswith(".pkl")]:
        results = {}
        print("Results for params file %s..." % params_fname)
        outdir = os.path.join(main_outdir, "results_%s" % params_fname)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        hpm_params = get_params(main_outdir, params_fname)

        print("Plot observation...")
        results["observation_rmse"] = plot_triple(
            None,  # data...
            partial(hpm_funcs["u_funcs"]["apply"], hpm_params["leaf"][LEAF].u, None),
            datasets,
            int(0.75 * cube_shape(datasets[LEAF].x)[2]),
            VMIN_OBS,
            VMAX_OBS,
            savefig=os.path.join(outdir, "observation"),
        )

        # print("Plot frames...")
        # plot_frames(hpm_funcs, hpm_params, datasets[LEAF], outdir)
        print("Plot physics...")
        results["physics_rmse"] = plot_triple(
            partial(hpm_funcs["apply_lhs"], hpm_params, leaf=LEAF),
            partial(hpm_funcs["apply_rhs"], hpm_params, leaf=LEAF),
            datasets,
            int(0.75 * cube_shape(datasets[LEAF].x)[2]),
            VMIN_PHYS,
            VMAX_PHYS,
            savefig=os.path.join(outdir, "physics"),
        )
        print("Plot crack...")
        plot_crack(hpm_funcs, hpm_params, datasets, outdir)

        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
