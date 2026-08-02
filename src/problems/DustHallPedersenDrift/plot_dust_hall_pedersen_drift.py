#!/usr/bin/env python3

"""Convert Hall-Pedersen drift CSV histories into a single-panel figure."""

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


HISTORY_FILE = "dust_hall_pedersen_drift_history.csv"
EXACT_FILE = "dust_hall_pedersen_drift_exact.csv"
OUTPUT_FILE = "dust_hall_pedersen_drift.pdf"


def legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], color="C0", marker="o", markerfacecolor="none", linestyle="None", label=r"$w_x$"),
        Line2D([], [], color="C1", marker="s", markerfacecolor="none", linestyle="None", label=r"$w_y$"),
    ]


def read_table(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return columns


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    history = read_table(data_dir / HISTORY_FILE)
    exact = read_table(data_dir / EXACT_FILE)

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH, 2.45))

    ax.plot(
        exact["t"],
        exact["wx_exact"],
        color="C0",
        linestyle="--",
    )
    ax.plot(
        history["t"],
        history["wx"],
        color="C0",
        marker="o",
        linestyle="None",
        zorder=3,
    )
    ax.plot(
        exact["t"],
        exact["wy_exact"],
        color="C1",
        linestyle="--",
    )
    ax.plot(
        history["t"],
        history["wy"],
        color="C1",
        marker="s",
        linestyle="None",
        zorder=3,
    )

    ax.set_xlabel("t")
    ax.set_ylabel(r"$w_x,\ w_y$")
    ax.set_xlim(0.0, 20.0)
    ax.legend(handles=legend_handles(), loc="center right", ncol=2)
    fig.tight_layout()

    output_path = output_dir / OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing the Hall-Pedersen drift CSV outputs.")
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
