#!/usr/bin/env python3

"""Plot the Hall-Pedersen drift history into a paper-style single-panel figure."""

from __future__ import annotations

import argparse
import csv
import os
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

PAPER_LABEL_FONTSIZE = 15
PAPER_TICK_FONTSIZE = 13
PAPER_TITLE_FONTSIZE = 14
PAPER_LEGEND_FONTSIZE = 12
MARKER_SIZE = 4.0
MARKER_EDGE_WIDTH = 1.0

plt.rcParams.update({
    "font.size": PAPER_TICK_FONTSIZE,
    "axes.labelsize": PAPER_LABEL_FONTSIZE,
    "axes.titlesize": PAPER_TITLE_FONTSIZE,
    "xtick.labelsize": PAPER_TICK_FONTSIZE,
    "ytick.labelsize": PAPER_TICK_FONTSIZE,
    "legend.fontsize": PAPER_LEGEND_FONTSIZE,
})


HISTORY_FILE = "dust_hall_pedersen_drift_history.csv"
EXACT_FILE = "dust_hall_pedersen_drift_exact.csv"
OUTPUT_FILE = "dust_hall_pedersen_drift.pdf"


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D([], [], color="C0", linewidth=1.2, label=r"$w_x$"),
        Line2D([], [], color="C1", linewidth=1.2, label=r"$w_y$"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="analytic"),
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
               markersize=MARKER_SIZE, label="simulation"),
    ]
    return handles


def read_table(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value) if value not in (None, "") else float("nan"))
    return columns


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    history = read_table(data_dir / HISTORY_FILE)
    exact = read_table(data_dir / EXACT_FILE)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))

    ax.plot(
        exact["t"],
        exact["wx_exact"],
        color="C0",
        linestyle="--",
        linewidth=1.2,
        label=r"analytic $w_x$",
    )
    ax.plot(
        history["t"],
        history["wx"],
        color="C0",
        marker="o",
        linestyle="None",
        markerfacecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=3,
    )
    ax.plot(
        exact["t"],
        exact["wy_exact"],
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label=r"analytic $w_y$",
    )
    ax.plot(
        history["t"],
        history["wy"],
        color="C1",
        marker="s",
        linestyle="None",
        markerfacecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
        markersize=MARKER_SIZE,
        zorder=3,
    )

    ax.set_xlabel("t")
    ax.set_ylabel(r"$w_x,\ w_y$")
    ax.legend(handles=legend_handles(), loc="best", frameon=False, ncol=2, columnspacing=1.0, handlelength=1.6)
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

    for filename in (HISTORY_FILE, EXACT_FILE):
        if not (data_dir / filename).exists():
            raise FileNotFoundError(f"Missing required CSV file: {filename}")

    output = make_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
