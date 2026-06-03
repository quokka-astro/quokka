#!/usr/bin/env python3

"""Plot the damped-gyromotion test cases into a 1x3 panel figure."""

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


OUTPUT_FILE = "dust_damped_gyromotion_panels.pdf"

SCHEMES = (
    ("tp2025", "TP2025", "C0", "o"),
    ("gl4", "GL4", "C1", "s"),
    ("midpoint", "Midpoint", "C2", "^"),
)

CASES = (
    ("pure_damping", "Pure Damping", r"$t/t_{s,0}$"),
    ("undamped_gyromotion", "Undamped Gyromotion", r"$\Omega_{\rm L} t$"),
    ("damped_gyromotion", "Damped Gyromotion", r"$t/t_{s,0}$"),
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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)

    for ax, (case_tag, title, xlabel) in zip(axes, CASES):
        history = read_table(data_dir / f"dust_damped_gyromotion_{case_tag}_history.csv")
        exact = read_table(data_dir / f"dust_damped_gyromotion_{case_tag}_exact.csv")
        show_legend = case_tag == CASES[0][0]

        ax.plot(
            exact["x_plot"],
            exact["wx_exact_norm"],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="analytic" if show_legend else "_nolegend_",
        )

        for slug, label, color, marker in SCHEMES:
            ax.plot(
                history["x_plot"],
                history[f"wx_{slug}_norm"],
                color=color,
                marker=marker,
                markersize=3.2,
                linestyle="None",
                label=label if show_legend else "_nolegend_",
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$w_x/w_0$")
        if show_legend:
            ax.legend(loc="best", frameon=False)

    fig.tight_layout()

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

    for case_tag, _, _ in CASES:
        for suffix in ("history", "exact"):
            filename = f"dust_damped_gyromotion_{case_tag}_{suffix}.csv"
            if not (data_dir / filename).exists():
                raise FileNotFoundError(f"Missing required CSV file: {filename}")

    output = make_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
