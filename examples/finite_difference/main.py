# File: main.py
# File Created: Tuesday, 7th July 2020 9:43:48 am
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Finite difference solver for wave equation
"""

import abc
import json
import os
from argparse import ArgumentParser
from collections import namedtuple
from time import time

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from tqdm import tqdm

from bhpm.util import plot_triple, timestamp

DIM = 2  # spatial dimension of the problem

torch.manual_seed(42)
# torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Utilities ============================================================================


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="If provided, load model from specified outdir instead of training a new one",
    )
    parser.add_argument(
        "--crack",
        action="store_true",
        help="Use the crack case (overridden if outdir is provided)",
    )

    return parser.parse_args()


def ensure_config(args):
    if args.outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "output", timestamp())
        crack = args.crack
        new_model = True
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
            with open(os.path.join(outdir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
    else:
        outdir = args.outdir
        if not os.path.isdir(outdir):
            raise RuntimeError("Failed to find specified run at %s" % outdir)
        with open(os.path.join(outdir, "args.json"), "r") as f:
            crack = json.load(f)["crack"]
        new_model = not os.path.isfile(os.path.join(outdir, "solver.pt"))

    return outdir, crack, new_model


def squared_distance(x1, x2):
    return (
        torch.power(x1, 2).sum(dim=1, keepdim=True)
        - 2.0 * x1 @ x2.T
        + torch.power(x2, 2).sum(dim=1, keepdim=True).T
    )


def _reverse(x):
    """
    Reverse a torch array since [::-1] isn't allowed by the current API
    """
    return x[torch.arange(len(x) - 1, -1, -1)]


class Elementwise(nn.Module):
    def __init__(self, f):
        super().__init__()
        self._f = f

    def forward(self, x):
        return self._f(x)


# Data =================================================================================

_us_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ultrasound_data")
)
_data_info = (
    # 0: Grains
    {
        "filename": os.path.abspath(
            os.path.join(_us_dir, "Hsample SAW 5MHz n2", "wvf.mat",)
        ),
        "crop": ((180, 230), (100, 150), (370, None)),  # T,Y,X
    },
    # 1: Crack, 0 degrees
    {
        "filename": os.path.abspath(
            os.path.join(
                _us_dir,
                "30Jan15 Nist crack1 240x240 12x12mm avg20 5MHz 0deg grips",
                "wvf.mat",
            )
        ),
        "crop": ((None, None), (None, None), (None, None)),
    },
)


def _load_data(case, verbose=False):
    """
    Load data cube from a file

    :return: namedtuple "Data" with np.ndarrays w/ following shapes:
        * x (NX,)
        * y (NY,)
        * t (NT,)
        * wavefield (NT, NY, NX)

    """
    filename = _data_info[case]["filename"]
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "Failed to find ultrasound data at %s.\nHave you downloaded it from Box?"
            % filename
        )

    wvf_mat = np.array(loadmat(filename)["wvf"])[:200]  # Stored as (NY,NX,NT)
    wvf = np.transpose(wvf_mat, (2, 0, 1))  # (NT,NY,NX)
    # crop:
    crop = _data_info[case]["crop"]
    wvf = wvf[crop[0][0] : crop[0][1], crop[1][0] : crop[1][1], crop[2][0] : crop[2][1]]
    nt, ny, nx = wvf.shape
    if verbose:
        print("Loaded %i measurements after cropping" % wvf.size)

    # Get coordinates
    xy_scale = 0.05  # Units: mm
    t_scale = 1.0 / 5.0  # Units: us (5 MHz sampling in time)
    x = xy_scale * np.arange(nx)
    y = xy_scale * np.arange(ny)
    t = t_scale * np.arange(nt)

    return namedtuple("_Data", ("x", "y", "t", "wvf"))(x, y, t, wvf)


def get_data(crack=False):
    """
    :return: (NT,NY,NX)
    """

    if crack:
        data = _load_data(1)
        wvf = data.wvf[570:630, 90:150, 70:160]  # 60 x 60 x 90
        rot = False
    else:
        data = _load_data(0)
        wvf = data.wvf  # 50 x 50 x 30
        # Rotate so source comes from below, not the right
        wvf = np.transpose(wvf[:, ::-1], (0, 2, 1)).copy()  # 50 x 30 x 50
        rot = True

    return namedtuple("Data", ("wvf", "rot"))(torch.Tensor(wvf).to(device), rot)


# Solver ===============================================================================


class CField(nn.Module):
    """
    Parent class for speed of sound fields
    """

    def forward(self, x, y):
        return self._forward(self._tile_cfield_inputs(x, y)).reshape(
            (y.numel(), x.numel())
        )

    @staticmethod
    def _tile_cfield_inputs(x, y):
        rev_y = _reverse(y)
        nx, ny = x.numel(), y.numel()
        x_ = torch.stack([x for _ in range(ny)])
        y_ = torch.stack([rev_y for _ in range(nx)]).T
        xy = torch.stack((x_.flatten(), y_.flatten())).T
        return xy

    @abc.abstractmethod
    def _forward(self, xy):
        """
        :param xy: (NY*NX, 2)
        :return: (NY*NX, 1)
        """
        raise NotImplementedError()


class CFieldConstant(CField):
    def __init__(self):
        super().__init__()
        self._raw_c = nn.Parameter(torch.tensor(0.0))
        self._c_transform = torch.distributions.transforms.ExpTransform()

    @property
    def c(self):
        return self._c_transform(self._raw_c)

    def _forward(self, xy):
        return self.c + 0.0 * xy[:, [0]]


class CFieldNet(CField):
    def __init__(self, units=64, layers=5):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(2, units),
            Elementwise(torch.sin),
            *([nn.Linear(units, units), Elementwise(torch.sin)] * (layers - 1)),
            nn.Linear(units, 1),
            Elementwise(torch.exp),
        )

        # Init tweaks
        self._net._modules["0"].weight.data = self._net._modules["0"].weight.data * 10.0
        for i in range(1, len(self._net._modules)):
            istr = str(i)
            if hasattr(self._net._modules[istr], "bias"):
                self._net._modules[istr].bias.data = (
                    self._net._modules[istr].bias.data * 0.0
                )

    def _forward(self, xy):
        return self._net(xy)


def get_solver_params(data, t_edge, x_edge, y_edge, t_oversample, s_oversample):
    """
    Reparameterize in terms of things that aare relative to the data being analyzed
    """
    nt_data, ny_data, nx_data = data.wvf.shape
    dt = 0.02 / t_oversample
    h = 0.05 / s_oversample
    window_stride = (t_oversample, s_oversample, s_oversample)
    window_corner = (
        t_edge * t_oversample,
        y_edge * s_oversample,
        x_edge * s_oversample,
    )
    nt = nt_data * t_oversample + window_corner[0] + 2
    ny = s_oversample * (ny_data + y_edge + 2)
    nx = s_oversample * (nx_data + 2 * x_edge)
    return {
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "dt": dt,
        "h": h,
        "window_corner": window_corner,
        "window_stride": window_stride,
        "data": data,
    }


class Solver(nn.Module):
    """
    Spatial units: mm
    temporal units: usec
    """

    def __init__(
        self,
        nx=240,
        ny=140,
        nt=5000,
        dt=0.005,  # Data is 0.2 per
        h=0.05,  # Data is 0.05 per
        window_corner=None,
        window_stride=None,
        data=None,
    ):
        super().__init__()
        self._h = h
        self._dt = dt

        self._x = nn.Parameter(h * torch.arange(nx), requires_grad=False)
        self._y = nn.Parameter(h * torch.arange(ny), requires_grad=False)
        self._t = nn.Parameter(dt * torch.arange(-2, nt), requires_grad=False)
        # T,Y,X
        self._window_corner = (30, 20, 30) if window_corner is None else window_corner
        self._window_stride = (40, 2, 2) if window_stride is None else window_stride

        # self.c_field = CFieldConstant()
        self.c_field = CFieldNet()

        # Source f(x,t) across the whole bottom of the simulation domain
        units, layers = 32, 5
        self.source = nn.Sequential(
            nn.Linear(2, units),
            Elementwise(torch.sin),
            *([nn.Linear(units, units), Elementwise(torch.sin)] * (layers - 1)),
            nn.Linear(units, 1),
        )

        # Apply physics via convolution
        self.step_kernel = nn.Conv2d(
            1, 1, 3, bias=False, padding=1, padding_mode="replicate"
        )
        self.step_kernel.requires_grad_(False)
        # Laplacian kernel:
        #  0  1  0
        #  1 -4  1
        #  0  1  0
        self.step_kernel.weight.data = torch.Tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        )[None][None]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def t(self):
        return self._t

    @property
    def nx(self):
        return len(self.x)

    @property
    def ny(self):
        return len(self.y)

    @property
    def nt(self):
        # Account for the two "dummy" input for the two IC slices.
        return len(self.t) - 2

    @property
    def window_corner(self):
        return self._window_corner

    @property
    def window_stride(self):
        return self._window_stride

    def forward(self):
        return self.simulate()

    def simulate(self, verbose=False, saveas=None):
        """
        :return: (NT, NY, NX)
        """
        c = self.c_field(self.x, self.y)
        coef = (self._dt * c / self._h) ** 2
        source_f = self.source(self._source_points()).reshape((self.nt + 2, self.nx))

        def step(u, u_prev, f):
            """
            perform a time step
            """
            u_step = coef * self.step_kernel(u[None][None])[0, 0]
            u_next = self._apply_source(-u_prev + 2.0 * u + u_step, f)
            return u_next

        u_list = self._initial_condition(source_f[:2])
        if verbose:
            print("Solve...")
            f_list = tqdm(source_f[2:])
        else:
            f_list = source_f[2:]
        for f in f_list:
            u_list.append(step(u_list[-1], u_list[-2], f))
        u = torch.stack(u_list[2:])  # NT, NY, NX

        # Visualize
        if saveas is not None:
            print("Save .npy...")
            np.save(saveas + ".npy", u.detach().cpu().numpy())
            print("Animate...")
            fig = plt.figure()
            ax = fig.gca()
            camera = Camera(fig)
            for ui in tqdm(u.detach().cpu().numpy()[::5]):
                ax.imshow(ui, vmin=-0.3, vmax=0.3)  # , cmap="bone")
                camera.snap()
            animation = camera.animate(interval=1)
            animation.save(saveas + ".gif")
            plt.close()
            print("Done!")

        return u

    def apply_window(self, u):
        c, s = self._window_corner, self._window_stride
        return u[c[0] :: s[0], c[1] :: s[1], c[2] :: s[2]]

    def to_data(self, data):
        nt, ny, nx = data.wvf.shape
        return self.apply_window(self.simulate())[:nt, :ny, :nx]

    def loss(self, data):
        """
        Assume dense measurement data for the moment
        """
        u = self.to_data(data)
        if not all([s_sim == s_data for s_sim, s_data in zip(u.shape, data.wvf.shape)]):
            msg = (
                "Simulation window can't match data (probably too small).\n"
                + "Simulation shape : "
                + str(u.shape)
                + "\n"
                + "Data shape       : "
                + str(data.wvf.shape)
            )
            raise ValueError(msg)
        return nn.MSELoss()(u, data.wvf)

    def _source_points(self):
        """
        :return: (x, t), shape ((NT+2)*NX, 2)
        """
        x_tiled = torch.stack([self.x for _ in range(self.nt + 2)])
        t_tiled = torch.stack([self.t for _ in range(self.nx)]).T
        return torch.stack((x_tiled.flatten(), t_tiled.flatten())).T

    def _initial_condition(self, source_f):
        """
        TODO complete IC field u0(x,y) instead of initializing at zero?

        :param source_f: (2, NX)
        """
        return [
            self._apply_source(torch.zeros(self.ny, self.nx).to(fi.device), fi)
            for fi in source_f
        ]

    def _apply_source(self, u, f):
        """
        :param u: (NY,NX)
        :param f: (NX,)
        """
        u[-1] = f
        u[-2] = f
        return u


# Inference ============================================================================


def train_from_scratch(data, outdir):
    iters = 10000
    lr_start = 3.0e-3
    lr_end = 3.0e-4

    solver = Solver(**get_solver_params(data, 10, 10, 10, 40, 2)).to(device)
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, iters, eta_min=lr_end
    )
    animate(solver, data, outdir, i=0)
    plot_crack(solver, outdir, i=0, wvf=data.wvf, rot=data.rot)
    losses = []
    t0 = time()
    for i in range(1, iters + 1):
        losses.append(train(solver, optimizer, data))
        print(
            "t = %6i | Iteration %6i / %6i | Loss = %.2e"
            % (time() - t0, i, iters, losses[-1])
        )
        if i % 100 == 0:
            torch.save(
                solver.state_dict(), os.path.join(outdir, "solver_ckpt_%i.pt" % i)
            )
            animate(solver, data, outdir, i=i)
            plot_crack(solver, outdir, i=i, wvf=data.wvf, rot=data.rot)
        scheduler.step()
    torch.save(solver.state_dict(), os.path.join(outdir, "solver.pt"))

    plt.figure()
    plt.semilogy(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    [plt.savefig(os.path.join(outdir, "loss." + ft)) for ft in ("png", "pdf")]
    plt.close()

    return solver


def load_solver(data, filename):
    solver = Solver(**get_solver_params(data, 10, 10, 10, 40, 2)).to(device)
    solver.load_state_dict(torch.load(filename))
    return solver


def train(solver, optimizer, data):
    solver.train()
    optimizer.zero_grad()
    loss = solver.loss(data)
    loss.backward()
    optimizer.step()
    return loss.item()


# Visualization ========================================================================


def animate(solver, data, outdir, i=None):
    """
    Make an animation to show how we did.
    """
    filename = "animation.gif" if i is None else ("animation_iter_%i.gif" % i)
    fig = plt.figure(figsize=(8, 4))
    ax_solver = fig.add_subplot(1, 2, 1)
    ax_data = fig.add_subplot(1, 2, 2)
    camera = Camera(fig)

    def _plot(ax, u, title=None, rot=False):
        if rot:
            # Unrotate
            u = u.T[::-1]
        ax.imshow(u, vmin=-10.0, vmax=10.0)
        ax.set_xticks(())
        ax.set_yticks(())
        if title is not None:
            ax.set_title(title)

    with torch.no_grad():
        for u_sol, u_data in zip(
            solver.to_data(data).detach().cpu().numpy(), data.wvf.detach().cpu().numpy()
        ):
            _plot(ax_solver, u_sol, "Solver", rot=data.rot)
            _plot(ax_data, u_data, "Data", rot=data.rot)
            camera.snap()
    animation = camera.animate()
    print("Saving animation...")
    animation.save(os.path.join(outdir, filename))
    print("Done!")
    plt.close()


def plot_crack(solver, outdir, i=None, wvf=None, rot=False):
    """
    :param wvf: If provided, clip to it
    """
    filebase = "crack" if i is None else ("crack_%i" % i)
    with torch.no_grad():
        c = solver.c_field(solver.x, solver.y).detach().cpu().numpy()
        if wvf is not None:
            # Don't subsample
            c = c[
                solver._window_corner[1] : solver._window_stride[1] * wvf.shape[1],
                solver._window_corner[2] : solver._window_stride[2] * wvf.shape[2],
            ]
        if rot:
            c = c.T[::-1]
        plt.figure()
        plt.imshow(c, vmin=0.0)
        plt.axis("off")
        plt.colorbar()
        [
            plt.savefig(os.path.join(outdir, filebase + ft), bbox_inches="tight")
            for ft in (".png", ".pdf")
        ]
        plt.close()


def plot_observed(solver, data, idx, outdir):
    """
    Compare simulation to data at a specific time index `idx` and save the figs.
    """
    with torch.no_grad():
        pred = solver.to_data(data)[idx].detach().cpu().numpy()
        targets = data.wvf[idx].detach().cpu().numpy()
    if data.rot:
        pred, targets = [z.T[::-1] for z in (pred, targets)]
    vmin, vmax = -10.0, 10.0
    plot_triple(targets, pred, vmin, vmax, saveas=os.path.join(outdir, "observation"))


# ======================================================================================


def main(args):
    outdir, crack, new_model = ensure_config(args)
    data = get_data(crack=crack)
    if new_model:
        solver = train_from_scratch(data, outdir)
    else:
        solver = load_solver(data, os.path.join(outdir, "solver.pt"))
    animate(solver, data, outdir)
    print("Plotting figures...")
    plot_observed(solver, data, int(0.75 * data.wvf.shape[0]), outdir)
    plot_crack(solver, outdir, wvf=data.wvf, rot=data.rot)
    print("...Done!")


if __name__ == "__main__":
    main(parse_args())
