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

SCHEMES = (
    ("tp2025", "TP2025", "C0", "o"),
    ("gl4", "GL4", "C1", "s"),
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

    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 3.0)
    if show_legend:
        ax.legend(handles=legend_handles(), loc="best")


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    history = read_table(data_dir / HISTORY_FILE)
    exact = read_table(data_dir / EXACT_FILE)

    fig, axes = plt.subplots(4, 1, figsize=(SINGLE_COLUMN_WIDTH, 6.1), sharex=True, gridspec_kw={"hspace": 0.0})
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

    for filename in (HISTORY_FILE, EXACT_FILE):
        if not (data_dir / filename).exists():
            raise FileNotFoundError(f"Missing required CSV file: {filename}")

    output = make_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
