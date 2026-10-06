#!/usr/bin/env python3

"""Plot multifluid and tracer diagnostics for DustyAlfvenWave."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from collections.abc import Callable
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
FIGURE_WIDTH = DOUBLE_COLUMN_WIDTH + 0.35

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


EPSILON_CASES = (
    ("epsilon1", r"$\epsilon=1$"),
    ("epsilon0p1", r"$\epsilon=0.1$"),
    ("epsilon0", r"$\epsilon=0$"),
)

OMEGA_CASES = (
    ("omega_high", r"$-\Omega_L/\Omega_{\rm AW}=10$"),
    ("omega_resonant", r"$-\Omega_L/\Omega_{\rm AW}=1$"),
    ("omega_low", r"$-\Omega_L/\Omega_{\rm AW}=0.1$"),
)

EPSILON_TOP_LIMITS = ((-0.1, 0.1), (-0.1, 0.1), (-0.5, 0.5))
EPSILON_BOTTOM_LIMITS = ((-0.58, 0.58),) * 3
OMEGA_LIMITS = ((-0.5, 0.5),) * 3

CasePlotter = Callable[[plt.Axes, plt.Axes, Path, str, str], None]


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value) if value not in ("", None) else float("nan"))
    return columns


def plot_multifluid_case(profile_ax: plt.Axes, history_ax: plt.Axes, data_dir: Path, sweep: str, tag: str) -> None:
    profile = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_profile.csv")
    history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_history.csv")

    profile_ax.plot(profile["z"], profile["ref_gas_vx"], color="tab:red", label="gas")
    profile_ax.plot(profile["z"], profile["ref_dust_vx"], color="black", label="dust")
    profile_ax.plot(profile["z"], profile["gas_vx"], color="tab:red", linestyle="None", marker="s", markevery=4)
    profile_ax.plot(profile["z"], profile["dust_vx"], color="black", linestyle="None", marker="s", markevery=4)
    history_ax.plot(history["t"], history["ref_dust_vx"], color="black")
    history_ax.plot(history["t"], history["dust_vx"], color="black", linestyle="None", marker="s", markevery=2)


def plot_tracer_case(profile_ax: plt.Axes, history_ax: plt.Axes, data_dir: Path, sweep: str, tag: str) -> None:
    profile = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_profile.csv")
    history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_history.csv")
    dense_history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_history_dense.csv")

    profile_ax.plot(profile["z_ref"], profile["gas_vx_ref"], color="tab:red", label="gas")
    profile_ax.plot(profile["z_ref"], profile["dust_vx_ref"], color="black", label="dust")
    profile_ax.plot(profile["z_num"], profile["gas_vx_num"], color="tab:red", linestyle="None", marker="s", markevery=4)
    profile_ax.plot(profile["z_num"], profile["dust_vx_num"], color="black", linestyle="None", marker="s", markevery=4)
    history_ax.plot(dense_history["t"], dense_history["ref_dust_vx"], color="black")
    history_ax.plot(history["t"], history["dust_vx"], color="black", linestyle="None", marker="s", markevery=2)


def make_figure(
    data_dir: Path,
    output_dir: Path,
    sweep: str,
    cases: tuple[tuple[str, str], ...],
    plot_case: CasePlotter,
    filename: str,
    top_row_limits: tuple[tuple[float, float], ...],
    bottom_row_limits: tuple[tuple[float, float], ...],
) -> Path:
    fig, axes = plt.subplots(2, len(cases), figsize=(FIGURE_WIDTH, 4.35), sharex="row")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.90, wspace=0.24, hspace=0.28)

    for column, (tag, title) in enumerate(cases):
        profile_ax = axes[0, column]
        history_ax = axes[1, column]
        plot_case(profile_ax, history_ax, data_dir, sweep, tag)
        profile_ax.set(xlim=(0.0, 1.0), ylim=top_row_limits[column], title=title)
        history_ax.set(xlim=(0.0, 5.0), ylim=bottom_row_limits[column])
        profile_ax.set_xlabel(r"$z$", labelpad=1.5)
        history_ax.set_xlabel(r"$t$", labelpad=1.0)

    axes[0, 0].set_ylabel(r"$v_x$")
    axes[1, 0].set_ylabel(r"$v_{d,x}$")
    axes[0, 0].legend(loc="best")

    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_comparison_figure(data_dir: Path, output_dir: Path) -> Path:
    """Compare Eulerian dust and tracer diagnostics at the same physical parameters."""
    fig, axes = plt.subplots(3, len(OMEGA_CASES), figsize=(FIGURE_WIDTH, 6.2), sharex="row")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.08, top=0.94, wspace=0.24, hspace=0.38)
    markers = {"linestyle": "None", "marker": "s", "markersize": 3.0, "markeredgewidth": 0.7, "zorder": 2}
    annotation_box = {"boxstyle": "round,pad=0.2", "facecolor": "0.92", "edgecolor": "none"}

    for column, (tag, title) in enumerate(OMEGA_CASES):
        profile_ax, fluid_ax, tracer_ax = axes[:, column]
        prefix = f"dusty_alfven_omega_{tag}"
        profile = read_csv(data_dir / f"{prefix}_profile.csv")
        history = read_csv(data_dir / f"{prefix}_history.csv")
        particles = read_csv(data_dir / f"{prefix}_particle_profile.csv")
        tracer_history = read_csv(data_dir / f"{prefix}_particle_history.csv")
        tracer_reference = read_csv(data_dir / f"{prefix}_particle_history_dense.csv")
        sample_z = history["z"][0]
        if any(z != sample_z for z in history["z"]) or sample_z != profile["z"][0]:
            raise ValueError(f"{prefix}: Eulerian history must sample the first cell.")

        # Draw references first, then markers, in gas / fluid / tracer order.
        profile_ax.plot(particles["z_ref"], particles["gas_vx_ref"], color="tab:red", label="gas", zorder=1)
        profile_ax.plot(profile["z"], profile["ref_dust_vx"], color="tab:blue", label="dust fluid", zorder=1)
        profile_ax.plot(particles["z_ref"], particles["dust_vx_ref"], color="black", label="dust tracer", zorder=1)
        profile_ax.plot(particles["z_num"], particles["gas_vx_num"], color="tab:red", markevery=4, **markers)
        profile_ax.plot(profile["z"], profile["dust_vx"], color="tab:blue", **markers)
        profile_ax.plot(particles["z_num"], particles["dust_vx_num"], color="black", markevery=4, **markers)
        fluid_ax.plot(history["t"], history["ref_dust_vx"], color="tab:blue", zorder=1)
        fluid_ax.plot(history["t"], history["dust_vx"], color="tab:blue", markevery=1, **markers)
        tracer_ax.plot(tracer_reference["t"], tracer_reference["ref_dust_vx"], color="black", zorder=1)
        tracer_ax.plot(tracer_history["t"], tracer_history["dust_vx"], color="black", markevery=1, **markers)

        profile_ax.set(xlim=(0.0, 1.0), ylim=OMEGA_LIMITS[column], title=title, xlabel=r"$z$")
        for ax in (fluid_ax, tracer_ax):
            ax.set(xlim=(0.0, 5.0), ylim=OMEGA_LIMITS[column], xlabel=r"$t$")
        tracer_ax.set_ylim(-0.55, 0.55)  # Include the resonant minimum below -0.5.
        if column == 0:
            for ax, label in (
                (fluid_ax, rf"$z=\Delta z/2={sample_z:g}$"),
                (tracer_ax, r"$z_0=0$"),
            ):
                ax.text(0.04, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=annotation_box)

    axes[0, 0].set_ylabel("x-velocity")
    axes[1, 0].set_ylabel("dust fluid x-velocity")
    axes[2, 0].set_ylabel("dust tracer x-velocity")
    axes[0, 0].legend(loc="upper right")

    output_path = output_dir / "dusty_alfven_omega_comparison.pdf"
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dusty_alfven_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        make_comparison_figure(data_dir, output_dir),
        make_figure(
            data_dir,
            output_dir,
            "epsilon",
            EPSILON_CASES,
            plot_multifluid_case,
            "dusty_alfven_epsilon_multifluid.pdf",
            EPSILON_TOP_LIMITS,
            EPSILON_BOTTOM_LIMITS,
        ),
        make_figure(
            data_dir,
            output_dir,
            "omega",
            OMEGA_CASES,
            plot_multifluid_case,
            "dusty_alfven_omega_multifluid.pdf",
            OMEGA_LIMITS,
            OMEGA_LIMITS,
        ),
        make_figure(
            data_dir,
            output_dir,
            "epsilon",
            EPSILON_CASES,
            plot_tracer_case,
            "dusty_alfven_epsilon_tracer.pdf",
            EPSILON_TOP_LIMITS,
            EPSILON_BOTTOM_LIMITS,
        ),
        make_figure(
            data_dir,
            output_dir,
            "omega",
            OMEGA_CASES,
            plot_tracer_case,
            "dusty_alfven_omega_tracer.pdf",
            OMEGA_LIMITS,
            OMEGA_LIMITS,
        ),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
