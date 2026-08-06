#!/usr/bin/env python3

"""Convert damped-gyromotion CSV histories into a 1x3 panel figure."""

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
        Line2D([], [], color=color, marker=marker, markerfacecolor="none", linestyle="None", label=label)
        for _, label, color, marker in SCHEMES
    )
    return handles


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

    output = make_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
