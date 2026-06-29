#!/usr/bin/env python3

"""Post-process DustLorentzShock CSV profiles into figure PDFs.

Run this script from the repository root or from ``tests/`` after generating the
CSV files with the ``DustLorentzShock`` executable.
"""

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


CASE_FILES = {
    "eps001_omega_ts0": "dust_lorentz_shock_eps001_omega_ts0.csv",
    "eps001_omega_ts20": "dust_lorentz_shock_eps001_omega_ts20.csv",
    "eps010_omega_ts20": "dust_lorentz_shock_eps010_omega_ts20.csv",
}

REGRESSION_CASES = ("eps001_omega_ts0", "eps001_omega_ts20", "eps010_omega_ts20")


def read_profile(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                if value is None or value == "":
                    columns[key].append(float("nan"))
                else:
                    columns[key].append(float(value))
    return columns


def has_required_files(data_dir: Path, case_names: tuple[str, ...]) -> bool:
    return all((data_dir / CASE_FILES[case_name]).exists() for case_name in case_names)


def plot_velocity_panel(
    ax,
    profile: dict[str, list[float]],
    title: str,
    guiding_key: str | None = None,
    show_legend: bool = False,
) -> None:
    dust_line, = ax.plot(profile["x"], profile["v_dx"], color="black", linewidth=1.3)
    gas_line, = ax.plot(profile["x"], profile["v_gx"], color="red", linewidth=1.1)
    guiding_line = None
    if guiding_key is not None and guiding_key in profile:
        guiding_line, = ax.plot(profile["x"], profile[guiding_key], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.6, 1.0)
    ax.set_title(title)
    if show_legend:
        handles = [gas_line, dust_line]
        labels = ["gas", "dust"]
        if guiding_line is not None:
            handles.append(guiding_line)
            labels.append("guiding-center")
        ax.legend(handles, labels, loc="best", frameon=False)


def plot_density_panel(ax, profile: dict[str, list[float]]) -> None:
    ax.plot(profile["x"], profile["rho_d_scaled"], color="black", linewidth=1.3)
    ax.plot(profile["x"], profile["rho_g"], color="red", linewidth=1.1)
    ax.set_xlim(0.6, 1.0)


def make_regression_figure(data_dir: Path, output_dir: Path) -> Path:
    shock_eps001_omega_ts0 = read_profile(data_dir / CASE_FILES["eps001_omega_ts0"])
    shock_eps001_omega_ts20 = read_profile(data_dir / CASE_FILES["eps001_omega_ts20"])
    shock_eps010_omega_ts20 = read_profile(data_dir / CASE_FILES["eps010_omega_ts20"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex="col")

    plot_velocity_panel(
        axes[0, 0],
        shock_eps001_omega_ts0,
        "$\\epsilon = 0.01,\\ \\Omega_{\\rm L} t_{\\rm s} = 0$",
    )
    axes[0, 0].set_ylabel(r"$v_x$")

    plot_velocity_panel(
        axes[0, 1],
        shock_eps010_omega_ts20,
        "$\\epsilon = 0.10,\\ \\Omega_{\\rm L} t_{\\rm s} = 20$",
        guiding_key="v_guiding_x",
    )

    plot_velocity_panel(
        axes[0, 2],
        shock_eps001_omega_ts20,
        "$\\epsilon = 0.01,\\ \\Omega_{\\rm L} t_{\\rm s} = 20$",
        guiding_key="v_guiding_x",
        show_legend=True,
    )

    plot_density_panel(axes[1, 0], shock_eps001_omega_ts0)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("density")

    plot_density_panel(axes[1, 1], shock_eps010_omega_ts20)
    axes[1, 1].set_xlabel("x")

    plot_density_panel(axes[1, 2], shock_eps001_omega_ts20)
    axes[1, 2].set_xlabel("x")

    fig.tight_layout()
    output_path = output_dir / "dust_lorentz_shock_regression.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dust_lorentz_shock_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not has_required_files(data_dir, REGRESSION_CASES):
        raise FileNotFoundError("Missing one or more DustLorentzShock CSV files.")

    output = make_regression_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
