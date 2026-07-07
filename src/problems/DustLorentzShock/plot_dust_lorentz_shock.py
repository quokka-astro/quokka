#!/usr/bin/env python3

"""Convert DustLorentzShock CSV profiles into a 2x3 panel figure."""

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


CASE_FILES = {
    "eps001_omega_ts0": "dust_lorentz_shock_eps001_omega_ts0.csv",
    "eps001_omega_ts20": "dust_lorentz_shock_eps001_omega_ts20.csv",
    "eps010_omega_ts20": "dust_lorentz_shock_eps010_omega_ts20.csv",
}

REGRESSION_CASES = ("eps001_omega_ts0", "eps001_omega_ts20", "eps010_omega_ts20")


def read_profile(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value) if value not in (None, "") else float("nan"))
    return columns


def has_required_files(data_dir: Path, case_names: tuple[str, ...]) -> bool:
    return all((data_dir / CASE_FILES[case_name]).exists() for case_name in case_names)


def plot_velocity_panel(
    ax: plt.Axes,
    profile: dict[str, list[float]],
    title: str,
    guiding_key: str | None = None,
    *,
    show_legend: bool = False,
) -> None:
    dust_line, = ax.plot(profile["x"], profile["v_dx"], color="black")
    gas_line, = ax.plot(profile["x"], profile["v_gx"], color="red")
    guiding_line = None
    if guiding_key is not None and guiding_key in profile:
        guiding_line, = ax.plot(profile["x"], profile[guiding_key], color="black", linestyle="--")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.0, 4.0)
    ax.set_title(title, pad=5.0)
    if show_legend:
        handles = [gas_line, dust_line]
        labels = ["gas", "dust"]
        if guiding_line is not None:
            handles.append(guiding_line)
            labels.append("guiding-center")
        ax.legend(handles, labels, loc="best")


def plot_density_panel(ax: plt.Axes, profile: dict[str, list[float]]) -> None:
    ax.plot(profile["x"], profile["rho_d_scaled"], color="black")
    ax.plot(profile["x"], profile["rho_g"], color="red")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.0, 6.0)


def make_regression_figure(data_dir: Path, output_dir: Path) -> Path:
    shock_eps001_omega_ts0 = read_profile(data_dir / CASE_FILES["eps001_omega_ts0"])
    shock_eps001_omega_ts20 = read_profile(data_dir / CASE_FILES["eps001_omega_ts20"])
    shock_eps010_omega_ts20 = read_profile(data_dir / CASE_FILES["eps010_omega_ts20"])

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH, 4.05), sharex="col")

    plot_velocity_panel(
        axes[0, 0],
        shock_eps001_omega_ts0,
        "$\\epsilon = 0.01,\\ \\Omega_{\\rm L} t_{\\rm s} = 0$",
    )
    axes[0, 0].set_ylabel(r"$v_x$")

    plot_velocity_panel(
        axes[0, 1],
        shock_eps010_omega_ts20,
        "$\\epsilon = 0.10,\\ \\Omega_{\\rm L} t_{\\rm s} = 20$",
        guiding_key="v_guiding_x",
    )

    plot_velocity_panel(
        axes[0, 2],
        shock_eps001_omega_ts20,
        "$\\epsilon = 0.01,\\ \\Omega_{\\rm L} t_{\\rm s} = 20$",
        guiding_key="v_guiding_x",
        show_legend=True,
    )

    plot_density_panel(axes[1, 0], shock_eps001_omega_ts0)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("density (scaled)")

    plot_density_panel(axes[1, 1], shock_eps010_omega_ts20)
    axes[1, 1].set_xlabel("x")

    plot_density_panel(axes[1, 2], shock_eps001_omega_ts20)
    axes[1, 2].set_xlabel("x")

    fig.tight_layout()
    output_path = output_dir / "dust_lorentz_shock_regression.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dust_lorentz_shock_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not has_required_files(data_dir, REGRESSION_CASES):
        raise FileNotFoundError("Missing one or more DustLorentzShock CSV files.")

    output = make_regression_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
