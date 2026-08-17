#!/usr/bin/env python3

"""Convert forced Hall-Pedersen diagnostic CSV data into two single-panel figures."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
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


DATA_FILE = "dust_forced_diagnostics.csv"
THEORY_DATA_FILE = "dust_forced_diagnostics_theory.csv"
OUTPUT_FILE = "dust_forced_diagnostics_panels.pdf"
REFERENCE_OUTPUT_FILE = "dust_forced_diagnostics_residual_reference.pdf"

SCHEMES = (
    ("gl4", "GL4", "C1", "s"),
    ("tp2025", "TP2025", "C0", "o"),
    ("midpoint", "Midpoint", "C2", "^"),
)


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | str] = {"scheme": row["scheme"]}
            for key, value in row.items():
                if key == "scheme":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def group_by_scheme(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scheme"])].append(row)
    for scheme_rows in grouped.values():
        scheme_rows.sort(key=lambda item: float(item["requested_dt"]))
    return grouped


def legend_handles() -> list[Line2D]:
    scheme_handles = [
        Line2D([], [], color=color, marker=marker, markerfacecolor="none", linestyle="None", label=label)
        for _, label, color, marker in SCHEMES
    ]
    return [Line2D([], [], color="black", linestyle="--", label="analytic")] + scheme_handles


def plot_panel(
    ax: plt.Axes,
    numerical_grouped: dict[str, list[dict[str, float | str]]],
    theory_grouped: dict[str, list[dict[str, float | str]]],
    value_key: str,
    theory_key: str,
    ylabel: str,
) -> None:
    for slug, _, color, marker in SCHEMES:
        numerical_rows = numerical_grouped[slug]
        requested_dt = [float(row["requested_dt"]) for row in numerical_rows]
        plot_floor = float(numerical_rows[0]["plot_floor"])
        values = [max(float(row[value_key]), plot_floor) for row in numerical_rows]
        ax.plot(
            requested_dt,
            values,
            color=color,
            marker=marker,
            markerfacecolor="none",
            linestyle="None",
            zorder=3,
        )

        theory_rows = theory_grouped[slug]
        for used_resolved_branch in (1.0, 0.0):
            branch_rows = [row for row in theory_rows if float(row["used_resolved_branch"]) == used_resolved_branch]
            theory_dt = [float(row["requested_dt"]) for row in branch_rows]
            theory_values = [max(float(row[theory_key]), plot_floor) for row in branch_rows]
            ax.plot(theory_dt, theory_values, color="black", linestyle="--", zorder=2)

    boundary_dt = float(numerical_grouped[SCHEMES[0][0]][0]["resolved_stiff_boundary_dt"])
    ax.axvline(boundary_dt, color="black", linestyle=":", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10.0**exponent for exponent in range(-3, 4)])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles(), loc="best")


def make_panel_figure(
    numerical_grouped: dict[str, list[dict[str, float | str]]],
    theory_grouped: dict[str, list[dict[str, float | str]]],
    output_path: Path,
    value_key: str,
    theory_key: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COLUMN_WIDTH, 2.35))
    plot_panel(ax, numerical_grouped, theory_grouped, value_key, theory_key, ylabel)
    ax.set_xlabel(r"$\Delta t$")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_figure(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    numerical_grouped = group_by_scheme(read_rows(data_dir / DATA_FILE))
    theory_grouped = group_by_scheme(read_rows(data_dir / THEORY_DATA_FILE))
    output_path = make_panel_figure(
        numerical_grouped,
        theory_grouped,
        output_dir / OUTPUT_FILE,
        "final_data_error",
        "terminal_error",
        r"distance to $\boldsymbol{w}_*$",
    )
    reference_output_path = make_panel_figure(
        numerical_grouped,
        numerical_grouped,
        output_dir / REFERENCE_OUTPUT_FILE,
        "final_to_fixed_point_error",
        "predicted_final_to_fixed_point_error",
        r"distance to $\boldsymbol{w}_{\rm fp}$",
    )
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

    output, reference_output = make_figure(data_dir, output_dir)
    print(output)
    print(reference_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
