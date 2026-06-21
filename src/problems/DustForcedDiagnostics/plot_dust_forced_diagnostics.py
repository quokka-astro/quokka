#!/usr/bin/env python3

"""Plot the forced Hall-Pedersen timestep diagnostics for the paper and a residual reference figure."""

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


DATA_FILE = "dust_forced_diagnostics.csv"
OUTPUT_FILE = "dust_forced_diagnostics_panels.pdf"
REFERENCE_OUTPUT_FILE = "dust_forced_diagnostics_residual_reference.pdf"

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


def paper_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
        )
        for _, label, color, marker in SCHEMES
    ]
    handles.extend((
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
               markersize=MARKER_SIZE, label=r"$\boldsymbol{w}_{\rm final}$"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.0, label=r"$\boldsymbol{w}_{\rm fp}$"),
    ))
    return handles


def reference_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=MARKER_SIZE,
            label=label,
        )
        for _, label, color, marker in SCHEMES
    ]
    handles.extend((
        Line2D([], [], color="black", marker="o", linestyle="None", markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
               markersize=MARKER_SIZE, label=r"$\boldsymbol{w}_{\rm final}$"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.0, label=r"predicted $\boldsymbol{w}_{\rm final}$"),
    ))
    return handles


def plot_panel(
    ax: plt.Axes,
    grouped: dict[str, list[dict[str, float | str]]],
    value_key: str,
    theory_key: str,
    ylabel: str,
    *,
    legend_handles: list[Line2D] | None = None,
) -> None:
    boundary_dt = None

    for slug, label, color, marker in SCHEMES:
        rows = grouped.get(slug, [])
        if not rows:
            continue

        requested_dt = [float(row["requested_dt"]) for row in rows]
        values = [max(float(row[value_key]), float(row["plot_floor"])) for row in rows]
        theory_values = [max(float(row[theory_key]), float(row["plot_floor"])) for row in rows]
        boundary_dt = float(rows[0]["resolved_stiff_boundary_dt"])

        ax.plot(
            requested_dt,
            values,
            color=color,
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=MARKER_SIZE,
            label="_nolegend_",
            zorder=3,
        )
        ax.plot(
            requested_dt,
            theory_values,
            color=color,
            linestyle="--",
            linewidth=1.0,
            label="_nolegend_",
            zorder=2,
        )

    if boundary_dt is not None:
        ax.axvline(boundary_dt, color="black", linestyle=":", linewidth=1.0, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    if legend_handles is not None:
        ax.legend(handles=legend_handles, loc="best", frameon=False)


def make_figure(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    grouped = group_by_scheme(read_rows(data_dir / DATA_FILE))

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.4))

    plot_panel(
        ax,
        grouped,
        "final_data_error",
        "terminal_error",
        r"distance to $\boldsymbol{w}_*$",
        legend_handles=paper_legend_handles(),
    )

    ax.set_xlabel(r"$\Delta t$")
    fig.tight_layout()

    output_path = output_dir / OUTPUT_FILE
    fig.savefig(output_path)
    plt.close(fig)

    ref_fig, ref_ax = plt.subplots(1, 1, figsize=(7.0, 4.4))
    plot_panel(
        ref_ax,
        grouped,
        "final_to_fixed_point_error",
        "predicted_final_to_fixed_point_error",
        r"distance to $\boldsymbol{w}_{\rm fp}$",
        legend_handles=reference_legend_handles(),
    )
    ref_ax.set_xlabel(r"$\Delta t$")
    ref_fig.tight_layout()

    reference_output_path = output_dir / REFERENCE_OUTPUT_FILE
    ref_fig.savefig(reference_output_path)
    plt.close(ref_fig)
    return output_path, reference_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dust_forced_diagnostics.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for the output PDF.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / DATA_FILE).exists():
        raise FileNotFoundError(f"Missing required CSV file: {DATA_FILE}")

    output, reference_output = make_figure(data_dir, output_dir)
    print(output)
    print(reference_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
