# File: util.py
# Created Date: 2020-06-26
# Author: Steven Atkinson (steven@atkinson.mn)

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def timestamp():
    t = datetime.now()
    return "%04i-%02i-%02i-%02i-%02i-%02i" % (
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second,
    )


def valid_timestamp(path):
    if not os.path.isdir(path):
        return False
    s = os.path.basename(path)
    vals = s.split("-")
    if len(vals) != 6:
        return False
    val_lengths = (4, 2, 2, 2, 2, 2)  # YYYY-MM-DD-HH-mm-SS
    for val, val_length in zip(vals, val_lengths):
        if len(val) != val_length:
            return False
        try:
            int(val)
        except ValueError:
            return False
    return True


def plot_triple(z_lhs: np.ndarray, z_rhs: np.ndarray, vmin, vmax, saveas=None):
    ny, nx = z_lhs.shape
    aspect = nx / ny
    fig = plt.figure(figsize=(7.0, 2.0 / aspect - 0.2))
    ax_lhs = fig.add_subplot(1, 5, 1)
    ax_lhs.imshow(z_lhs, vmin=vmin, vmax=vmax)  # , extent=extent)
    ax_lhs.axis("off")

    ax_rhs = fig.add_subplot(1, 5, 2)
    s1 = ax_rhs.imshow(z_rhs, vmin=vmin, vmax=vmax)  # , extent=extent)
    ax_rhs.axis("off")

    cax1 = fig.add_subplot(1, 5, 3)
    plt.colorbar(s1, cax=cax1)

    ax_err = fig.add_subplot(1, 5, 4)
    s2 = ax_err.imshow(np.log10(np.abs(z_lhs - z_rhs)))  # , extent=extent)
    ax_err.axis("off")

    cax2 = fig.add_subplot(1, 5, 5)
    plt.colorbar(s2, cax=cax2)

    subfig_width = 0.25
    cax_width = 0.01
    ax_lhs.set_position((0.0, 0.0, subfig_width, 1.0))
    ax_rhs.set_position((0.26, 0.0, subfig_width, 1.0))
    cax1.set_position((0.52, 0.0, cax_width, 1.0))
    ax_err.set_position((0.62, 0.0, subfig_width, 1.0))
    cax2.set_position((0.88, 0.0, cax_width, 1.0))

    if saveas is not None:
        [plt.savefig(saveas + ft, bbox_inches="tight") for ft in (".png", ".pdf")]
        plt.close()
    else:
        plt.show()
