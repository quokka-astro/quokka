#!/usr/bin/env python3

"""Post-process DustyOrszagTang CSV diagnostics into figure PDFs."""

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
import numpy as np


CASE_INFO = {
    "high_mu": {"label": r"$\mu \approx 0.45$", "title": r"$\mu \approx 0.45$"},
    "low_mu": {"label": r"$\mu \approx 4.5\times10^{-6}$", "title": r"$\mu \approx 4.5\times10^{-6}$"},
}
SNAPSHOTS = ("t0p25", "t0p50")


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value is not None and value != ""})
    return rows


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


def make_fig6(data_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True, sharey=True, constrained_layout=True)
    contour_levels = [0.9, 1.0, 1.1, 1.2, 1.35, 1.5]

    for row, case_tag in enumerate(("high_mu", "low_mu")):
        for col, snapshot in enumerate(SNAPSHOTS):
            rows = read_csv(data_dir / f"dusty_orszag_tang_{case_tag}_{snapshot}_slice.csv")
            xs, ys, rho_g, rho_d_scaled = reshape_slice(rows)
            ax = axes[row, col]
            mesh = ax.pcolormesh(xs, ys, rho_g, shading="nearest", cmap="magma")
            normalized_dust = rho_d_scaled / np.mean(rho_d_scaled)
            ax.contour(xs, ys, normalized_dust, levels=contour_levels, colors="black", linewidths=0.55, alpha=0.75)
            if row == 0:
                ax.set_title(f"t = {0.25 if snapshot == 't0p25' else 0.5:g}")
            if col == 0:
                ax.set_ylabel(f"y\n{CASE_INFO[case_tag]['label']}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")

    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\rho_g$")
    output_path = output_dir / "dusty_orszag_tang_fig6_analog.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_fig7(data_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, constrained_layout=True)

    for col, case_tag in enumerate(("high_mu", "low_mu")):
        rows = read_csv(data_dir / f"dusty_orszag_tang_{case_tag}_t0p25_profile.csv")
        y = np.array([row["y"] for row in rows])
        mask = y <= 0.3 + 1.0e-12
        y = y[mask]
        rho_g = np.array([row["rho_g"] for row in rows])[mask]
        rho_d_scaled = np.array([row["rho_d_scaled"] for row in rows])[mask]
        v_gy = np.array([row["v_gy"] for row in rows])[mask]
        v_dy = np.array([row["v_dy"] for row in rows])[mask]

        axes[0, col].plot(y, v_dy, color="black", linewidth=1.3, label="dust")
        axes[0, col].plot(y, v_gy, color="red", linewidth=1.1, label="gas")
        axes[0, col].set_title(CASE_INFO[case_tag]["title"])
        axes[0, col].set_ylabel(r"$v_y$")

        axes[1, col].plot(y, rho_d_scaled, color="black", linewidth=1.3, label="dust")
        axes[1, col].plot(y, rho_g, color="red", linewidth=1.1, label="gas")
        axes[1, col].set_ylabel("density")
        axes[1, col].set_xlabel("y")

    axes[0, 0].legend(frameon=False, loc="upper right")
    axes[1, 0].legend(frameon=False, loc="upper right")
    for ax in axes.flat:
        ax.set_xlim(0.0, 0.3)

    output_path = output_dir / "dusty_orszag_tang_fig7_analog.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.cwd(), help="Directory containing dusty_orszag_tang_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Directory for output PDFs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig6 = make_fig6(data_dir, output_dir)
    fig7 = make_fig7(data_dir, output_dir)
    print(fig6)
    print(fig7)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
