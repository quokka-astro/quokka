#!/usr/bin/env python3

"""Convert damped-gyromotion CSV histories into diagnostic figures."""

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
from matplotlib.lines import Line2D

DOUBLE_COLUMN_WIDTH = 6.9
SINGLE_COLUMN_WIDTH = 3.4
TRAJECTORY_TICK_SCALE = 0.7
TRAJECTORY_MARKER_SCALE = 0.8
TRAJECTORY_ANNOTATION_SIZE = 8.0
TRAJECTORY_TICK_PAD = 0.0

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


OUTPUT_FILE = "dust_damped_gyromotion_panels.pdf"
DYNAMIC_TRAJECTORIES_OUTPUT_FILE = "dust_dynamic_coefficient_trajectories.pdf"
DYNAMIC_ERRORS_OUTPUT_FILE = "dust_dynamic_coefficient_errors.pdf"

SCHEMES = (
    ("gl4", "GL4", "C1", "s"),
    ("tp2025", "TP2025", "C0", "o"),
    ("midpoint", "Midpoint", "C2", "^"),
)

CASES = (
    ("pure_damping", r"$\Omega_{\rm L} t_{\rm s,0}=0$", r"$t/t_{s,0}$", 2.0),
    ("undamped_gyromotion", r"$\Omega_{\rm L} t_{\rm s,0}\to\infty$", r"$\Omega_{\rm L} t$", 10.0),
    ("damped_gyromotion", r"$\Omega_{\rm L} t_{\rm s,0}=5$", r"$t/t_{s,0}$", 2.0),
)

TREATMENTS = (
    ("frozen", "Frozen", "C4", "D", 3),
    ("picard", "Picard", "C1", "s", 4),
)


def read_table(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value) if value not in (None, "") else float("nan"))
    return columns


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def legend_handles() -> list[Line2D]:
    handles = [Line2D([], [], color="black", linestyle="--", label="analytic")]
    handles.extend(
        Line2D([], [], color=color, marker=marker, markerfacecolor="none", linestyle="None", label=label)
        for _, label, color, marker in SCHEMES
    )
    return handles


def coefficient_treatment_legend_handles() -> tuple[Line2D, ...]:
    treatments = tuple(
        Line2D([], [], color=color, marker=marker, markerfacecolor="none", linestyle="None", label=label)
        for _, label, color, marker, _ in TREATMENTS
    )
    return (Line2D([], [], color="black", linestyle="--", label="analytic"),) + treatments


def error_legend_handles() -> tuple[tuple[Line2D, ...], tuple[Line2D, ...]]:
    treatments = tuple(
        Line2D([], [], color=color, marker=marker, markerfacecolor="none", linestyle="None", label=label)
        for _, label, color, marker, _ in TREATMENTS
    )
    coefficient_models = (
        Line2D([], [], color="black", linestyle="-", label=r"$t_{\rm s}(W),\ \xi=1$"),
        Line2D([], [], color="black", linestyle="--", label=r"$t_{\rm s}(W),\ \xi(W)$"),
    )
    return treatments, coefficient_models


def plot_dynamic_panel(
    ax: plt.Axes,
    exact_x: list[float],
    exact_y: list[float],
    numerical_x: list[float],
    numerical_y: tuple[list[float], ...],
) -> None:
    ax.plot(exact_x, exact_y, color="black", linestyle="--", zorder=2)
    for values, (_, _, color, marker, zorder) in zip(numerical_y, TREATMENTS):
        ax.plot(
            numerical_x,
            values,
            color=color,
            marker=marker,
            markerfacecolor="none",
            markeredgewidth=TRAJECTORY_MARKER_SCALE * plt.rcParams["lines.markeredgewidth"],
            markersize=TRAJECTORY_MARKER_SCALE * plt.rcParams["lines.markersize"],
            linestyle="None",
            zorder=zorder,
        )


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 2.55), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.88, wspace=0.17)

    for ax, (case_tag, title, xlabel, xmax) in zip(axes, CASES):
        history = read_table(data_dir / f"dust_damped_gyromotion_{case_tag}_history.csv")
        exact = read_table(data_dir / f"dust_damped_gyromotion_{case_tag}_exact.csv")

        ax.plot(exact["x_plot"], exact["wx_exact_norm"], color="black", linestyle="--", zorder=2)

        for slug, _, color, marker in SCHEMES:
            ax.plot(
                history["x_plot"],
                history[f"wx_{slug}_norm"],
                color=color,
                marker=marker,
                linestyle="None",
                zorder=3,
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlim(0.0, xmax)

    axes[0].set_ylabel(r"$w_x/w_0$")
    axes[0].legend(handles=legend_handles(), loc="best")

    output_path = output_dir / OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_dynamic_trajectory_figure(data_dir: Path, output_dir: Path) -> Path:
    epstein = read_table(data_dir / "dust_dynamic_epstein_iteration_history.csv")
    charge = read_table(data_dir / "dust_dynamic_charge_iteration_history.csv")
    charge_exact = read_table(data_dir / "dust_dynamic_charge_iteration_exact.csv")

    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_COLUMN_WIDTH, 3.6), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.98, hspace=0.12, wspace=0.34)

    plot_dynamic_panel(
        axes[0, 0],
        epstein["t"],
        epstein["wx_exact_norm"],
        epstein["t"],
        (epstein["wx_gl4_frozen_norm"], epstein["wx_gl4_picard_norm"]),
    )
    plot_dynamic_panel(
        axes[0, 1],
        epstein["t"],
        epstein["wy_exact_norm"],
        epstein["t"],
        (epstein["wy_gl4_frozen_norm"], epstein["wy_gl4_picard_norm"]),
    )
    plot_dynamic_panel(
        axes[1, 0],
        charge_exact["t"],
        charge_exact["wx_exact_norm"],
        charge["t"],
        (charge["wx_frozen_norm"], charge["wx_picard_norm"]),
    )
    plot_dynamic_panel(
        axes[1, 1],
        charge_exact["t"],
        charge_exact["xi_exact"],
        charge["t"],
        (charge["xi_frozen"], charge["xi_picard"]),
    )

    for ax in axes.flat:
        ax.set_xlim(0.0, 2.0)
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=TRAJECTORY_TICK_SCALE * plt.rcParams["xtick.labelsize"],
            pad=TRAJECTORY_TICK_PAD,
        )
        ax.tick_params(
            axis="both",
            which="major",
            length=TRAJECTORY_TICK_SCALE * plt.rcParams["xtick.major.size"],
            width=TRAJECTORY_TICK_SCALE * plt.rcParams["xtick.major.width"],
        )
        ax.tick_params(
            axis="both",
            which="minor",
            length=TRAJECTORY_TICK_SCALE * plt.rcParams["xtick.minor.size"],
            width=TRAJECTORY_TICK_SCALE * plt.rcParams["xtick.minor.width"],
        )
    axes[0, 0].set_ylim(-0.25, 1.05)
    axes[1, 0].set_ylim(-0.25, 1.05)
    axes[1, 1].set_ylim(-1.0, 0.1)

    axes[0, 0].set_ylabel(r"$w_x/w_0$", fontsize=TRAJECTORY_ANNOTATION_SIZE)
    axes[0, 1].set_ylabel(r"$w_y/w_0$", fontsize=TRAJECTORY_ANNOTATION_SIZE)
    axes[1, 0].set_ylabel(r"$w_x/w_0$", fontsize=TRAJECTORY_ANNOTATION_SIZE)
    axes[1, 1].set_ylabel(r"$\xi$", fontsize=TRAJECTORY_ANNOTATION_SIZE)
    axes[1, 0].set_xlabel(r"$t/t_{\rm s,0}$", fontsize=TRAJECTORY_ANNOTATION_SIZE)
    axes[1, 1].set_xlabel(r"$t/t_{\rm s,0}$", fontsize=TRAJECTORY_ANNOTATION_SIZE)

    for ax, label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        ax.text(0.96, 0.95, label, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    annotation_box = {"boxstyle": "round,pad=0.2", "facecolor": "0.92", "edgecolor": "none"}
    axes[0, 0].text(
        0.04,
        0.95,
        r"$t_{\rm s}(W),\ \xi=1$",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="top",
        fontsize=TRAJECTORY_ANNOTATION_SIZE,
        bbox=annotation_box,
    )
    axes[1, 0].text(
        0.04,
        0.95,
        r"$t_{\rm s}(W),\ \xi(W)$",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=TRAJECTORY_ANNOTATION_SIZE,
        bbox=annotation_box,
    )
    axes[0, 0].legend(
        handles=coefficient_treatment_legend_handles(),
        loc="center right",
        fontsize=TRAJECTORY_ANNOTATION_SIZE,
    )

    output_path = output_dir / DYNAMIC_TRAJECTORIES_OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_dynamic_error_figure(data_dir: Path, output_dir: Path) -> Path:
    epstein_rows = [row for row in read_rows(data_dir / "dust_dynamic_epstein_iteration_convergence.csv") if row["scheme"] == "gl4"]
    epstein = {key: [float(row[key]) for row in epstein_rows] for key in ("dt", "frozen_error", "picard_error")}
    charge = read_table(data_dir / "dust_dynamic_charge_convergence.csv")

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.6))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.98)

    for treatment, _, color, marker, zorder in TREATMENTS:
        ax.loglog(
            epstein["dt"],
            epstein[f"{treatment}_error"],
            color=color,
            marker=marker,
            markerfacecolor="none",
            linestyle="-",
            zorder=zorder,
        )
        ax.loglog(
            charge["dt"],
            charge[f"{treatment}_error"],
            color=color,
            marker=marker,
            markerfacecolor=color,
            linestyle="--",
            zorder=zorder,
        )

    ax.set_xlabel(r"$\Delta t/t_{\rm s,0}$")
    ax.set_ylabel(r"$e_w$")
    treatment_handles, coefficient_handles = error_legend_handles()
    treatment_legend = ax.legend(handles=treatment_handles, loc="center left")
    ax.add_artist(treatment_legend)
    ax.legend(handles=coefficient_handles, loc="lower right")

    output_path = output_dir / DYNAMIC_ERRORS_OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing the dust_damped_gyromotion CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for the output PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        make_figure(data_dir, output_dir),
        make_dynamic_trajectory_figure(data_dir, output_dir),
        make_dynamic_error_figure(data_dir, output_dir),
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
