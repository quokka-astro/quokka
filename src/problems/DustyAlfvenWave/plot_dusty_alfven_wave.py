#!/usr/bin/env python3

"""Post-process DustyAlfvenWave CSV files into Moseley-style panel figures."""

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


MU_CASES = (
    ("mu0", r"$\mu=0$"),
    ("mu0p01", r"$\mu=0.01$"),
    ("mu0p1", r"$\mu=0.1$"),
    ("mu1", r"$\mu=1$"),
)

OMEGA_CASES = (
    ("omega_high", r"$-\omega_L/\Omega_{\rm AW}=10$"),
    ("omega_resonant", r"$-\omega_L/\Omega_{\rm AW}=1$"),
    ("omega_low", r"$-\omega_L/\Omega_{\rm AW}=0.1$"),
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
        for kind in ("profile", "history"):
            path = data_dir / f"dusty_alfven_{sweep}_{tag}_{kind}.csv"
            if not path.exists():
                missing.append(path)
    if missing:
        names = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing DustyAlfvenWave CSV files:\n{names}")


def require_particle_files(data_dir: Path, sweep: str, cases: tuple[tuple[str, str], ...]) -> None:
    missing: list[Path] = []
    for tag, _ in cases:
        for kind in ("particle_profile", "particle_history", "particle_history_dense"):
            path = data_dir / f"dusty_alfven_{sweep}_{tag}_{kind}.csv"
            if not path.exists():
                missing.append(path)
    if missing:
        names = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing DustyAlfvenWave particle CSV files:\n{names}")


def plot_profile_panel(ax, profile: dict[str, list[float]], title: str) -> None:
    ax.plot(profile["z"], profile["ref_dust_vx"], color="black", linewidth=1.3, label="dust ref")
    ax.plot(profile["z"], profile["ref_gas_vx"], color="tab:red", linewidth=1.1, label="gas ref")
    ax.plot(profile["z"], profile["dust_vx"], color="black", linestyle="None", marker="s", markersize=3.0, label="dust")
    ax.plot(profile["z"], profile["gas_vx"], color="tab:red", linestyle="None", marker="o", markersize=3.0, label="gas")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)


def plot_history_panel(ax, history: dict[str, list[float]]) -> None:
    ax.plot(history["t"], history["ref_dust_vx"], color="black", linewidth=1.3)
    ax.plot(history["t"], history["dust_vx"], color="black", linestyle="None", marker="s", markersize=2.5)
    ax.set_xlim(0.0, 5.0)


def plot_particle_profile_panel(ax, profile: dict[str, list[float]], title: str) -> None:
    ax.plot(profile["z_ref"], profile["dust_vx_ref"], color="black", linewidth=1.3, label="dust ref")
    ax.plot(profile["z_ref"], profile["gas_vx_ref"], color="tab:red", linewidth=1.1, label="gas ref")
    ax.plot(profile["z_num"], profile["dust_vx_num"], color="black", linestyle="None", marker="s", markersize=3.0, label="dust")
    ax.plot(profile["z_num"], profile["gas_vx_num"], color="tab:red", linestyle="None", marker="o", markersize=3.0, label="gas")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)


def plot_particle_history_panel(ax, history: dict[str, list[float]], dense_history: dict[str, list[float]]) -> None:
    ax.plot(dense_history["t"], dense_history["ref_dust_vx"], color="black", linewidth=1.3)
    ax.plot(history["t"], history["dust_vx"], color="black", linestyle="None", marker="s", markersize=2.5)
    ax.set_xlim(0.0, 5.0)


def make_figure(data_dir: Path, output_dir: Path, sweep: str, cases: tuple[tuple[str, str], ...], filename: str) -> Path:
    require_files(data_dir, sweep, cases)

    fig, axes = plt.subplots(2, len(cases), figsize=(4.0 * len(cases), 7.0), sharex="row")
    if len(cases) == 1:
        axes = axes.reshape(2, 1)

    for column, (tag, title) in enumerate(cases):
        profile = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_profile.csv")
        history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_history.csv")

        plot_profile_panel(axes[0, column], profile, title)
        plot_history_panel(axes[1, column], history)
        axes[1, column].set_xlabel(r"$t$")

    axes[0, 0].set_ylabel(r"$x$ velocity at $t=5$")
    axes[1, 0].set_ylabel(r"$v_{d,x}$")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path = output_dir / filename
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_particle_figure(data_dir: Path, output_dir: Path, sweep: str, cases: tuple[tuple[str, str], ...], filename: str) -> Path:
    require_particle_files(data_dir, sweep, cases)

    fig, axes = plt.subplots(2, len(cases), figsize=(4.0 * len(cases), 7.0), sharex="row")
    if len(cases) == 1:
        axes = axes.reshape(2, 1)

    for column, (tag, title) in enumerate(cases):
        profile = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_profile.csv")
        history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_history.csv")
        dense_history = read_csv(data_dir / f"dusty_alfven_{sweep}_{tag}_particle_history_dense.csv")

        plot_particle_profile_panel(axes[0, column], profile, title)
        plot_particle_history_panel(axes[1, column], history, dense_history)
        axes[1, column].set_xlabel(r"$t$")

    axes[0, 0].set_ylabel(r"tracer $x$ velocity at $t=5$")
    axes[1, 0].set_ylabel(r"$v_{d,x}$")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
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
        make_figure(data_dir, output_dir, "mu", MU_CASES, "dusty_alfven_mu.pdf"),
        make_figure(data_dir, output_dir, "omega", OMEGA_CASES, "dusty_alfven_omega.pdf"),
        make_particle_figure(data_dir, output_dir, "mu", MU_CASES, "dusty_alfven_mu_paper_like.pdf"),
        make_particle_figure(data_dir, output_dir, "omega", OMEGA_CASES, "dusty_alfven_omega_paper_like.pdf"),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
