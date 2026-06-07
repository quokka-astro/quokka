#!/usr/bin/env python3

"""Plot the mixed stopping-time drag-damping test into a 1x4 panel figure."""

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

PAPER_LABEL_FONTSIZE = 15
PAPER_TICK_FONTSIZE = 13
PAPER_TITLE_FONTSIZE = 14
PAPER_LEGEND_FONTSIZE = 12

plt.rcParams.update({
    "font.size": PAPER_TICK_FONTSIZE,
    "axes.labelsize": PAPER_LABEL_FONTSIZE,
    "axes.titlesize": PAPER_TITLE_FONTSIZE,
    "xtick.labelsize": PAPER_TICK_FONTSIZE,
    "ytick.labelsize": PAPER_TICK_FONTSIZE,
    "legend.fontsize": PAPER_LEGEND_FONTSIZE,
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


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    history = read_table(data_dir / HISTORY_FILE)
    exact = read_table(data_dir / EXACT_FILE)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharex=False)

    for ax, (prefix, ylabel) in zip(axes, PANELS):
        show_legend = prefix == PANELS[0][0]
        analytic_handle = ax.plot(
            exact["t"],
            exact[f"{prefix}_exact"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="analytic" if show_legend else "_nolegend_",
        )[0]

        for slug, label, color, marker in SCHEMES:
            ax.plot(
                history["t"],
                history[f"{prefix}_{slug}"],
                color=color,
                marker=marker,
                markersize=3.2,
                linewidth=1.0,
                label=label if show_legend else "_nolegend_",
            )

        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        if show_legend:
            ax.legend(loc="best", frameon=False)

    fig.tight_layout()

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
