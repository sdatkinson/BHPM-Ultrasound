# File: train.py
# Created Date: 2020-06-25
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Train on cracked data with full model:

utt = a(x, y, u, uxx, uyy) f(u, ux, uy, uxx, uyy)
Note that wave equation is recovered when f = uxx + uyy

TODO:
* Fix variance for f~GP or else pathological 
    * via rbf() kernel -> rbf_fixedvariance()

HACKS:
* Pretrain on MSE not likelihood (won't work for BHPM)
* Using joint training on MSE loss type (OK for HPM, but not BHPM)
"""

from argparse import ArgumentParser
import json
import os
import pickle

import jax.config
from jax.random import PRNGKey, split
import matplotlib.pyplot as plt

from bhpm import ultrasound, util


def parse_args():
    parser = ArgumentParser()
    # Both
    parser.add_argument(
        # See bhpm.ultrasound.data, _info
        "--data-type",
        type=str,
        choices=("pristine", "cracked"),
        default="pristine",
        help="Which data to run",
    )
    parser.add_argument(
        # See bhpm.ultrasound.model, _structure_1, _structure_2
        "--model-type",
        type=str,
        choices=("ipinn", "bhpm"),
        default=1,
        help="Which model to use",
    )
    parser.add_argument(
        "--u-iters",
        type=int,
        default=50000,
        help="Number of iterations to train solution for",
    )
    parser.add_argument(
        "--af-iters",
        type=int,
        default=100000,
        help="Number of iterations to train solution for",
    )
    parser.add_argument(
        "--f-from",
        type=str,
        default=None,
        help='Load and freeze f physics from another run (provide timestamp or "latest")',
    )

    return parser.parse_args()


def print_args(outdir, args):
    with open(os.path.join(outdir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)


def get_physics(main_outdir, f_from):
    if f_from == "latest":
        # Don't use this one!
        rundir = sorted(
            [
                d
                for d in os.listdir(main_outdir)
                if util.valid_timestamp(os.path.join(main_outdir, d))
            ]
        )[-2]
    else:
        rundir = f_from
    filename = os.path.join(main_outdir, rundir, "params.pkl")
    if not os.path.isfile(filename):
        raise FileNotFoundError("Failed to find parameters at %s" % filename)
    print("Get global physics from %s" % filename)
    with open(filename, "rb") as f:
        return pickle.load(f)["root"]


def save_params(outdir, params, name2=""):
    with open(os.path.join(outdir, "params%s.pkl" % name2), "wb") as f:
        pickle.dump(params, f)


def main(args, testing=False):
    freeze_f = args.f_from is not None
    if args.model_type == "bhpm":
        print("NOTE: x64 for BHPM (GPs)...")
        jax.config.update("jax_enable_x64", True)

    rng = PRNGKey(42)
    main_outdir = os.path.join(os.path.dirname(__file__), "output")
    outdir = os.path.join(main_outdir, util.timestamp())
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print_args(outdir, args)
    rng, rng_data = split(rng)
    datasets = ultrasound.get_data(args.data_type)

    hpm_type = {"ipinn": ultrasound.hpm_1, "bhpm": ultrasound.hpm_2}[args.model_type]
    hpm_funcs = hpm_type(
        datasets, root_type={"ipinn": "wave", "bhpm": "gp"}[args.model_type]
    )
    # 1) Init
    rng, rng_init = split(rng)
    hpm_params = hpm_funcs["init"](rng_init)
    if args.f_from is not None:
        hpm_params["root"] = get_physics(main_outdir, args.f_from)
    # 2) Train u then af
    rng, rng_train = split(rng)
    hpm_params = hpm_funcs["train"](
        rng_train,
        hpm_params,
        datasets,
        u_iters=args.u_iters,
        af_iters=args.af_iters,
        freeze_f=freeze_f,
    )
    hpm_funcs["print_mses"](hpm_params, datasets)
    save_params(outdir, hpm_params)

    print("Done")
    plt.close("all")


if __name__ == "__main__":
    main(parse_args())
