#!/usr/bin/env python3

"""Convert mixed-stiff drag-damping CSV histories into a 4x1 panel figure."""

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
from matplotlib.ticker import NullLocator

SINGLE_COLUMN_WIDTH = 3.4

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


HISTORY_FILE = "dust_damping_mhd_zero_b_mixed_stiff_history.csv"
EXACT_FILE = "dust_damping_mhd_zero_b_mixed_stiff_exact.csv"
OUTPUT_FILE = "dust_damping_mhd_zero_b_mixed_stiff_panels.pdf"
SWEEP_FILE = "dust_damping_mhd_zero_b_mixed_stiff_timestep_sweep.csv"
SWEEP_OUTPUT_FILE = "dust_damping_mhd_zero_b_mixed_stiff_timestep_sweep.pdf"
T_MAX = 2.0

SCHEMES = (
    ("gl4", "GL4", "C1", "s"),
    ("tp2025", "TP2025", "C0", "o"),
    ("midpoint", "Midpoint", "C2", "^"),
)

PANELS = (
    ("v_gas", r"$v_g$"),
    ("v_dust1", r"$v_{d,1}$"),
    ("v_dust2", r"$v_{d,2}$"),
    ("E_gas", r"$E_g$"),
)


def read_table(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value) if value not in (None, "") else float("nan"))
    return columns


def legend_handles() -> list[Line2D]:
    handles = [Line2D([], [], color="black", linestyle="--", label="analytic")]
    handles.extend(
        Line2D([], [], color=color, marker=marker, markerfacecolor="none", linestyle="-", label=label)
        for _, label, color, marker in SCHEMES
    )
    return handles


def plot_panel(ax: plt.Axes, history: dict[str, list[float]], exact: dict[str, list[float]], prefix: str, ylabel: str, *, show_legend: bool) -> None:
    ax.plot(exact["t"], exact[f"{prefix}_exact"], color="black", linestyle="--", zorder=2)

    for slug, _, color, marker in SCHEMES:
        ax.plot(
            history["t"],
            history[f"{prefix}_{slug}"],
            color=color,
            marker=marker,
            linestyle="-",
            zorder=3,
        )

    if prefix == "E_gas":
        ax.plot(
            history["t"],
            history["E_gas_tp2025_no_residual_correction"],
            color="0.45",
            linestyle="-.",
            zorder=2,
            label="TP2025 (no residual correction)",
        )
        ax.legend(loc="best", fontsize=7.5)

    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, T_MAX)
    if show_legend:
        ax.legend(handles=legend_handles(), loc="best")


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    history = read_table(data_dir / HISTORY_FILE)
    exact = read_table(data_dir / EXACT_FILE)

    fig, axes = plt.subplots(4, 1, figsize=(SINGLE_COLUMN_WIDTH, 6.1), sharex=True)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.08, top=0.995, hspace=0.0)

    for index, (ax, (prefix, ylabel)) in enumerate(zip(axes, PANELS)):
        plot_panel(ax, history, exact, prefix, ylabel, show_legend=(index == 0))

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("t")

    output_path = output_dir / OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_timestep_sweep_figure(data_dir: Path, output_dir: Path) -> Path:
    with (data_dir / SWEEP_FILE).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH, 2.5))
    for slug, label, color, marker in SCHEMES:
        scheme_rows = [row for row in rows if row["scheme"] == slug]
        for used_resolved_branch in (1, 0):
            branch_rows = [row for row in scheme_rows if int(row["used_resolved_branch"]) == used_resolved_branch]
            ax.plot(
                [float(row["requested_dt"]) for row in branch_rows],
                [float(row["velocity_error"]) for row in branch_rows],
                color=color,
                marker=marker,
                linestyle="-",
                zorder=3,
            )

    fast_stopping_time = float(rows[0]["fast_stopping_time"])
    branch_transition_dt = float(rows[0]["branch_transition_dt"])
    ax.axvline(fast_stopping_time, color="0.45", linestyle="--", zorder=1)
    ax.axvline(branch_transition_dt, color="black", linestyle=":", zorder=1)
    ax.legend(handles=legend_handles()[1:], loc="upper right", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-4, 1.0e3)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("relative velocity error")
    ax.text(
        fast_stopping_time,
        0.04,
        r"$t_{{\rm s},2}$",
        color="0.35",
        rotation=90,
        va="bottom",
        ha="right",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        branch_transition_dt,
        0.04,
        r"$t_{{\rm s},1}$",
        rotation=90,
        va="bottom",
        ha="right",
        transform=ax.get_xaxis_transform(),
    )
    fig.tight_layout()

    output_path = output_dir / SWEEP_OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing the mixed-stiff CSV outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for the output PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output = make_figure(data_dir, output_dir)
    sweep_output = make_timestep_sweep_figure(data_dir, output_dir)
    print(output)
    print(sweep_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
