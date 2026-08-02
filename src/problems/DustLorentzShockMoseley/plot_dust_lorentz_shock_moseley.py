#!/usr/bin/env python3

"""Convert DustLorentzShockMoseley CSV profiles into a 2x3 panel figure."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "quokka-matplotlib-cache"
_mpl_config_dir = _cache_root / "mplconfig"
_xdg_cache_dir = _cache_root / "xdg-cache"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOUBLE_COLUMN_WIDTH = 6.9

_LATEX_AVAILABLE = shutil.which("latex") is not None

plt.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.45,
    "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.25,
    "legend.columnspacing": 0.7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.8,
    "lines.markerfacecolor": "none",
    "lines.markeredgewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.minor.size": 0.0,
    "ytick.minor.size": 0.0,
    "axes.formatter.use_mathtext": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

if _LATEX_AVAILABLE:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Latin Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
    })
else:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


CASES = (
    (
        "moseley_eps1em4_omega1p8_ts0p04",
        "$\\epsilon = 10^{-4},\\ \\Omega_{\\rm L} t_{\\rm s} = 1.8$\n$ t_{\\rm s} = 0.04\\,L/c_{\\rm s}$",
    ),
    (
        "moseley_eps1em1_omega3p0_ts0p04",
        "$\\epsilon = 10^{-1},\\ \\Omega_{\\rm L} t_{\\rm s} = 3.0$\n$ t_{\\rm s} = 0.04\\,L/c_{\\rm s}$",
    ),
    (
        "moseley_eps1em4_omega12_ts0p10",
        "$\\epsilon = 10^{-4},\\ \\Omega_{\\rm L} t_{\\rm s} = 12$\n$ t_{\\rm s} = 0.10\\,L/c_{\\rm s}$",
    ),
)

OUTPUT_FILE = "dust_lorentz_shock_moseley.pdf"


def read_profile(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return columns


def plot_velocity_panel(ax: plt.Axes, profile: dict[str, list[float]], title: str, *, show_legend: bool = False) -> None:
    dust_line, = ax.plot(profile["x"], profile["v_dx"], color="black")
    gas_line, = ax.plot(profile["x"], profile["v_gx"], color="red")
    guiding_line, = ax.plot(profile["x"], profile["v_guiding_x"], color="black", linestyle="--")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.0, 4.0)
    ax.set_title(title, pad=5.0)
    if show_legend:
        ax.legend(
            (gas_line, dust_line, guiding_line),
            ("gas", "dust", "guiding-center"),
            loc="best",
        )


def plot_density_panel(ax: plt.Axes, profile: dict[str, list[float]]) -> None:
    ax.plot(profile["x"], profile["rho_d_scaled"], color="black")
    ax.plot(profile["x"], profile["rho_g"], color="red")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.0, 6.0)


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH, 4.05), sharex="col")

    for column, (tag, title) in enumerate(CASES):
        profile = read_profile(data_dir / f"dust_lorentz_shock_{tag}.csv")
        plot_velocity_panel(axes[0, column], profile, title, show_legend=(column == 2))
        plot_density_panel(axes[1, column], profile)

    axes[0, 0].set_ylabel(r"$v_x$")
    axes[1, 0].set_ylabel("density (scaled)")
    for ax in axes[1]:
        ax.set_xlabel("x")

    fig.tight_layout()
    output_path = output_dir / OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing the DustLorentzShockMoseley CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for the output PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output = make_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
