#!/usr/bin/env python3

"""Convert DustyAlfvenWave CSV files into four 2x3 panel figures."""

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


EPSILON_CASES = (
    ("epsilon1", r"$\epsilon=1$"),
    ("epsilon0p1", r"$\epsilon=0.1$"),
    ("epsilon0", r"$\epsilon=0$"),
)

OMEGA_CASES = (
    ("omega_high", r"$-\Omega_L/\Omega_{\rm AW}=10$"),
    ("omega_resonant", r"$-\Omega_L/\Omega_{\rm AW}=1$"),
    ("omega_low", r"$-\Omega_L/\Omega_{\rm AW}=0.1$"),
)


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value) if value not in ("", None) else float("nan"))
    return columns


def require_files(data_dir: Path, sweep: str, cases: tuple[tuple[str, str], ...]) -> None:
    missing: list[Path] = []
    for tag, _ in cases:
        for kind in ("particle_profile", "particle_history", "particle_history_dense"):
            path = data_dir / f"dusty_alfven_{sweep}_{tag}_{kind}.csv"
            if not path.exists():
                missing.append(path)
    if missing:
        names = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing DustyAlfvenWave particle CSV files:\n{names}")


def plot_particle_profile_panel(ax: plt.Axes, profile: dict[str, list[float]], title: str) -> None:
    ax.plot(profile["z_ref"], profile["gas_vx_ref"], color="tab:red", label="gas")
    ax.plot(profile["z_ref"], profile["dust_vx_ref"], color="black", label="dust")
    ax.plot(profile["z_num"], profile["gas_vx_num"], color="tab:red", linestyle="None", marker="s", markevery=4)
    ax.plot(profile["z_num"], profile["dust_vx_num"], color="black", linestyle="None", marker="s", markevery=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)


def plot_particle_history_panel(ax: plt.Axes, history: dict[str, list[float]], dense_history: dict[str, list[float]]) -> None:
    ax.plot(dense_history["t"], dense_history["ref_dust_vx"], color="black")
    ax.plot(history["t"], history["dust_vx"], color="black", linestyle="None", marker="s", markevery=2)
    ax.set_xlim(0.0, 5.0)


def figure_width() -> float:
    return DOUBLE_COLUMN_WIDTH + 0.35


def make_figure(
    data_dir: Path,
    output_dir: Path,
    sweep: str,
    cases: tuple[tuple[str, str], ...],
    filename: str,
    y_limits: tuple[float, float] | None = None,
    top_row_limits: tuple[tuple[float, float], ...] | None = None,
    bottom_row_limits: tuple[tuple[float, float], ...] | None = None,
) -> Path:
    require_files(data_dir, sweep, cases)

    fig, axes = plt.subplots(2, len(cases), figsize=(figure_width(), 4.35), sharex="row")
    if len(cases) == 1:
        axes = axes.reshape(2, 1)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.90, wspace=0.24, hspace=0.28)

    for column, (tag, title) in enumerate(cases):
        profile = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_profile.csv")
        history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_history.csv")
        dense_history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_history_dense.csv")

        plot_particle_profile_panel(axes[0, column], profile, title)
        plot_particle_history_panel(axes[1, column], history, dense_history)
        axes[0, column].set_xlabel(r"$z$", labelpad=1.5)
        axes[1, column].set_xlabel(r"$t$", labelpad=1.0)
        if y_limits is not None:
            axes[0, column].set_ylim(*y_limits)
            axes[1, column].set_ylim(*y_limits)
        if top_row_limits is not None:
            axes[0, column].set_ylim(*top_row_limits[column])
        if bottom_row_limits is not None:
            axes[1, column].set_ylim(*bottom_row_limits[column])

    axes[0, 0].set_ylabel(r"$v_x$")
    axes[1, 0].set_ylabel(r"$v_{d,x}$")
    axes[0, 0].legend(loc="best")

    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dusty_alfven_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        make_figure(
            data_dir,
            output_dir,
            "epsilon",
            EPSILON_CASES,
            "dusty_alfven_epsilon.pdf",
            top_row_limits=((-0.1, 0.1), (-0.1, 0.1), (-0.5, 0.5)),
            bottom_row_limits=((-0.58, 0.58), (-0.58, 0.58), (-0.58, 0.58)),
        ),
        make_figure(data_dir, output_dir, "omega", OMEGA_CASES, "dusty_alfven_omega.pdf", y_limits=(-0.5, 0.5)),
        make_figure(
            data_dir,
            output_dir,
            "epsilon",
            EPSILON_CASES,
            "dusty_alfven_epsilon_paper_like.pdf",
            top_row_limits=((-0.1, 0.1), (-0.1, 0.1), (-0.5, 0.5)),
            bottom_row_limits=((-0.58, 0.58), (-0.58, 0.58), (-0.58, 0.58)),
        ),
        make_figure(data_dir, output_dir, "omega", OMEGA_CASES, "dusty_alfven_omega_paper_like.pdf", y_limits=(-0.5, 0.5)),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
