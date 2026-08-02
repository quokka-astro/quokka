#!/usr/bin/env python3

"""Convert DustyOrszagTangPaper CSV diagnostics into a 2x2 slice figure and a 2x1 profile figure."""

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
import numpy as np
from matplotlib.lines import Line2D

SINGLE_COLUMN_WIDTH = 3.4
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

OUTPUT_PREFIX = "dusty_orszag_tang_paper"

CASE_LABELS = {
    "high_epsilon": r"$\epsilon \approx 0.45$",
    "low_epsilon": r"$\epsilon \approx 4.5\times10^{-6}$",
}
SNAPSHOTS = (("t0p25", 0.25), ("t0p50", 0.5))
CONTOUR_LEVELS = [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1]
LINE_STYLES = {
    64: ":",
    128: (0, (6.0, 2.3, 1.4, 2.3)),
    256: "--",
    512: "-",
}


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def csv_path(data_dir: Path, prefix: str, resolution: int, case_tag: str, snapshot_tag: str, kind: str) -> Path:
    return data_dir / f"{prefix}_{resolution}_{case_tag}_{snapshot_tag}_{kind}.csv"


def reshape_slice(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array(sorted({row["x"] for row in rows}))
    ys = np.array(sorted({row["y"] for row in rows}))
    nx = xs.size
    ny = ys.size
    rho_g = np.empty((ny, nx))
    rho_d_scaled = np.empty((ny, nx))
    x_index = {value: idx for idx, value in enumerate(xs)}
    y_index = {value: idx for idx, value in enumerate(ys)}
    for row in rows:
        i = x_index[row["x"]]
        j = y_index[row["y"]]
        rho_g[j, i] = row["rho_g"]
        rho_d_scaled[j, i] = row["rho_d_scaled"]
    return xs, ys, rho_g, rho_d_scaled


def trim_shared_edge_ticks(ax: plt.Axes, *, axis: str, drop_first: bool = False, drop_last: bool = False) -> None:
    if axis == "x":
        ticks = ax.get_xticks()
        lo, hi = ax.get_xlim()
        setter = ax.set_xticks
    else:
        ticks = ax.get_yticks()
        lo, hi = ax.get_ylim()
        setter = ax.set_yticks

    span = max(abs(hi - lo), 1.0)
    tol = 1.0e-9 * span
    visible_ticks = [tick for tick in ticks if (lo - tol) <= tick <= (hi + tol)]
    trimmed_ticks = [
        tick for tick in visible_ticks if not ((drop_first and abs(tick - lo) <= tol) or (drop_last and abs(tick - hi) <= tol))
    ]
    setter(trimmed_ticks)


def make_fig6(data_dir: Path, output_dir: Path, prefix: str, resolution: int) -> Path:
    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 6.6))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.08),
        wspace=0.0,
        hspace=0.0,
        left=0.07,
        right=0.93,
        bottom=0.08,
        top=0.94,
    )
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = fig.add_subplot(grid[1, 0], sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 1] = fig.add_subplot(grid[1, 1], sharex=axes[0, 0], sharey=axes[0, 0])
    cax = fig.add_subplot(grid[:, 2])
    for row, case_tag in enumerate(("high_epsilon", "low_epsilon")):
        for col, (snapshot, time) in enumerate(SNAPSHOTS):
            rows = read_csv(csv_path(data_dir, prefix, resolution, case_tag, snapshot, "slice"))
            xs, ys, rho_g, rho_d_scaled = reshape_slice(rows)
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                xs,
                ys,
                rho_g,
                shading="nearest",
                cmap="magma",
                edgecolors="none",
                linewidth=0.0,
                antialiased=False,
                rasterized=True,
            )
            normalized_dust = rho_d_scaled / np.mean(rho_d_scaled)
            ax.contour(xs, ys, normalized_dust, levels=CONTOUR_LEVELS, colors="black", linewidths=0.55, alpha=0.75)
            if row == 0:
                ax.set_title(f"t = {time:g}")
            if col == 0:
                ax.text(
                    0.03,
                    0.97,
                    CASE_LABELS[case_tag],
                    color="white",
                    fontsize=11.0,
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            if row == 0:
                ax.tick_params(labelbottom=False)
            if col == 1:
                ax.tick_params(labelleft=False)

    fig.supxlabel("x")
    fig.supylabel("y")
    trim_shared_edge_ticks(axes[1, 0], axis="x", drop_last=True)
    trim_shared_edge_ticks(axes[1, 1], axis="x", drop_first=True)
    trim_shared_edge_ticks(axes[0, 0], axis="y", drop_first=True)
    trim_shared_edge_ticks(axes[1, 0], axis="y", drop_last=True)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\rho_g$")
    output_path = output_dir / "dusty_orszag_tang_paper_fig6_analog.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_fig7(data_dir: Path, output_dir: Path, prefix: str, case_tag: str, resolutions: list[int]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_WIDTH, 3.8), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.10, top=0.95, hspace=0.0)

    for resolution in resolutions:
        rows = read_csv(csv_path(data_dir, prefix, resolution, case_tag, "t0p25", "profile"))
        y = np.array([row["y"] for row in rows])
        mask = y <= 0.3 + 1.0e-12
        y = y[mask]
        rho_g = np.array([row["rho_g"] for row in rows])[mask]
        rho_d_scaled = np.array([row["rho_d_scaled"] for row in rows])[mask]
        v_gy = np.array([row["v_gy"] for row in rows])[mask]
        v_dy = np.array([row["v_dy"] for row in rows])[mask]
        linestyle = LINE_STYLES[resolution]

        axes[0].plot(y, v_dy, color="black", linestyle=linestyle)
        axes[0].plot(y, v_gy, color="red", linestyle=linestyle)
        axes[1].plot(y, rho_d_scaled, color="black", linestyle=linestyle)
        axes[1].plot(y, rho_g, color="red", linestyle=linestyle)

    axes[0].set_title(CASE_LABELS[case_tag])
    axes[0].set_ylabel(r"$v_y$")
    axes[1].set_ylabel("density")
    axes[1].set_xlabel("y")
    axes[1].set_ylim(bottom=0.0)

    legend_handles = [Line2D([0], [0], color="red", linestyle="-", label="gas")]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=LINE_STYLES[resolution],
            label=(rf"${resolution}^2$(dust)" if resolution == 512 else rf"${resolution}^2$"),
        )
        for resolution in sorted(resolutions, reverse=True)
    )
    axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        handlelength=1.9,
        labelspacing=0.2,
        borderaxespad=0.15,
    )

    axes[0].set_xlim(0.0, 0.3)

    axes[0].tick_params(labelbottom=False)
    output_path = output_dir / "dusty_orszag_tang_paper_fig7_analog.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dusty_orszag_tang_paper_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    parser.add_argument("--prefix", default=OUTPUT_PREFIX, help="CSV file prefix to read.")
    parser.add_argument("--fig6-resolution", type=int, default=512, help="Resolution tag used for the Fig. 6-style panel plot.")
    parser.add_argument(
        "--fig7-resolutions",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Resolution tags used for the Fig. 7-style convergence plot.",
    )
    parser.add_argument(
        "--fig7-case",
        choices=tuple(CASE_LABELS),
        default="high_epsilon",
        help="Case tag used for the Fig. 7-style convergence plot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig7_resolutions = sorted(set(args.fig7_resolutions))
    unsupported_resolutions = [resolution for resolution in fig7_resolutions if resolution not in LINE_STYLES]
    if unsupported_resolutions:
        raise ValueError(f"Unsupported fig7 resolutions: {unsupported_resolutions}. Expected a subset of {sorted(LINE_STYLES)}.")

    fig6 = make_fig6(data_dir, output_dir, args.prefix, args.fig6_resolution)
    fig7 = make_fig7(data_dir, output_dir, args.prefix, args.fig7_case, fig7_resolutions)
    print(fig6)
    print(fig7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
