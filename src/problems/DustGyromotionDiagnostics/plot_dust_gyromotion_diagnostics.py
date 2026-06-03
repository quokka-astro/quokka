#!/usr/bin/env python3

"""Plot the pure-gyromotion timestep diagnostics into a 2x1 panel figure."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
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


DATA_FILE = "dust_gyromotion_diagnostics.csv"
OUTPUT_FILE = "dust_gyromotion_diagnostics_panels.pdf"

SCHEMES = (
    ("tp2025", "TP2025", "C0", "o"),
    ("gl4", "GL4", "C1", "s"),
    ("midpoint", "Midpoint", "C2", "^"),
)


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | str] = {"scheme": row["scheme"] or ""}
            for key, value in row.items():
                if key == "scheme":
                    continue
                parsed[key] = float(value) if value not in (None, "") else float("nan")
            rows.append(parsed)
    return rows


def group_by_scheme(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scheme"])].append(row)
    for scheme_rows in grouped.values():
        scheme_rows.sort(key=lambda item: float(item["requested_dt"]))
    return grouped


def plot_panel(ax, grouped: dict[str, list[dict[str, float | str]]], value_key: str, theory_key: str, ylabel: str, *, show_legend: bool) -> None:
    boundary_dt = None

    for slug, label, color, marker in SCHEMES:
        rows = grouped.get(slug, [])
        if not rows:
            continue

        requested_dt = [float(row["requested_dt"]) for row in rows]
        values = [float(row[value_key]) for row in rows]
        theory_values = [max(abs(float(row[theory_key])), float(row["plot_floor"])) for row in rows]
        boundary_dt = float(rows[0]["resolved_stiff_boundary_dt"])

        ax.plot(
            requested_dt,
            values,
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=1.0,
            label=label if show_legend else "_nolegend_",
        )
        ax.plot(
            requested_dt,
            theory_values,
            color=color,
            linestyle="--",
            linewidth=1.0,
        )

    if boundary_dt is not None:
        ax.axvline(boundary_dt, color="black", linestyle=":", linewidth=1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(loc="best", frameon=False)


def make_figure(data_dir: Path, output_dir: Path) -> Path:
    grouped = group_by_scheme(read_rows(data_dir / DATA_FILE))

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.5), sharex=True)

    plot_panel(axes[0], grouped, "abs_delta_log_amplitude", "theory_delta_log_amplitude", r"$|\delta a|$", show_legend=True)
    plot_panel(axes[1], grouped, "abs_delta_phase", "theory_delta_phase", r"$|\delta \phi|$", show_legend=False)

    axes[1].set_xlabel(r"$\Delta t$")
    fig.tight_layout()

    output_path = output_dir / OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dust_gyromotion_diagnostics.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for the output PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / DATA_FILE).exists():
        raise FileNotFoundError(f"Missing required CSV file: {DATA_FILE}")

    output = make_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
