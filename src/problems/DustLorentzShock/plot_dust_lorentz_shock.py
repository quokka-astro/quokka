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


CASE_FILES = {
    "ref_neutral": "dust_lorentz_shock_ref_neutral.csv",
    "charged_dilute": "dust_lorentz_shock_charged_dilute.csv",
    "charged_backreacting": "dust_lorentz_shock_charged_backreacting.csv",
}

LOW_MACH_CASES = ("ref_neutral", "charged_dilute", "charged_backreacting")


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


def plot_velocity_panel(ax, profile: dict[str, list[float]], title: str, guiding_key: str | None = None, neutral_reference=None) -> None:
    ax.plot(profile["x"], profile["v_dx"], color="black", linewidth=1.3)
    ax.plot(profile["x"], profile["v_gx"], color="red", linewidth=1.1)
    if guiding_key is not None and guiding_key in profile:
        ax.plot(profile["x"], profile[guiding_key], color="black", linestyle="--", linewidth=1.0)
    if neutral_reference is not None:
        ax.plot(neutral_reference["x"], neutral_reference["v_dx"], color="black", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)


def plot_density_panel(ax, profile: dict[str, list[float]]) -> None:
    ax.plot(profile["x"], profile["rho_d_scaled"], color="black", linewidth=1.3)
    ax.plot(profile["x"], profile["rho_g"], color="red", linewidth=1.1)
    ax.set_xlim(0.0, 1.0)


def make_low_mach_figure(data_dir: Path, output_dir: Path) -> Path:
    ref_neutral = read_profile(data_dir / CASE_FILES["ref_neutral"])
    charged_dilute = read_profile(data_dir / CASE_FILES["charged_dilute"])
    charged_backreacting = read_profile(data_dir / CASE_FILES["charged_backreacting"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex="col")

    plot_velocity_panel(axes[0, 0], ref_neutral, "Neutral reference")
    axes[0, 0].set_ylabel("v_x")

    plot_velocity_panel(axes[0, 1], charged_backreacting, "Charged, mu = 0.10")

    plot_velocity_panel(
        axes[0, 2],
        charged_dilute,
        "Charged, mu = 0.01",
        guiding_key="v_guiding_x",
        neutral_reference=ref_neutral,
    )

    plot_density_panel(axes[1, 0], ref_neutral)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("density")

    plot_density_panel(axes[1, 1], charged_backreacting)
    axes[1, 1].set_xlabel("x")

    plot_density_panel(axes[1, 2], charged_dilute)
    axes[1, 2].set_xlabel("x")

    fig.tight_layout()
    output_path = output_dir / "dust_lorentz_shock_fig2_analog.pdf"
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

    if not has_required_files(data_dir, LOW_MACH_CASES):
        raise FileNotFoundError("Missing one or more DustLorentzShock CSV files.")

    output = make_low_mach_figure(data_dir, output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
