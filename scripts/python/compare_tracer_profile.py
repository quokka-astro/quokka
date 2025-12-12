#!/usr/bin/env python3
"""
Compare tracer deposition against the Eulerian density profile from
hydro_wave_tracer_profile.csv (written by HydroWaveTracerLong).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_profile(csv_path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def summarize(tracer_frac: np.ndarray, mass_frac: np.ndarray, rel_err: np.ndarray) -> str:
    l1 = float(np.sum(np.abs(tracer_frac - mass_frac)))
    l2 = float(np.sqrt(np.sum((tracer_frac - mass_frac) ** 2)))
    linf = float(np.max(np.abs(rel_err)))
    mean_rel = float(np.mean(rel_err))
    return (
        f"L1(tracer_frac - mass_frac) = {l1:.6e}\n"
        f"L2(tracer_frac - mass_frac) = {l2:.6e}\n"
        f"max(|relative_error|)       = {linf:.6e}\n"
        f"mean(relative_error)        = {mean_rel:.6e}"
    )


def plot_profiles(x: np.ndarray, tracer_frac: np.ndarray, mass_frac: np.ndarray, rel_err: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax0 = axes[0]
    ax0.plot(x, mass_frac, label="mass fraction", color="C0")
    ax0.plot(x, tracer_frac, label="tracer fraction", color="C1", linestyle="--")
    ax0.set_ylabel("Fraction")
    ax0.legend(loc="best")
    ax0.set_title("Tracer vs mass fractions")

    ax1 = axes[1]
    ax1.plot(x, rel_err, label="(tracer_frac / mass_frac) - 1", color="C3")
    ax1.axhline(0.0, color="k", linewidth=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Relative error")
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tracer deposition to density profile.")
    parser.add_argument("--csv", type=Path, default=Path("hydro_wave_tracer_profile.csv"), help="Path to hydro_wave_tracer_profile.csv")
    parser.add_argument("--out", type=Path, default=Path("tracer_profile_comparison.png"), help="Output plot path")
    args = parser.parse_args()

    profile = load_profile(args.csv)
    x = profile["x"]
    tracer_frac = profile["tracer_fraction"]
    mass_frac = profile["mass_fraction"]
    rel_err = profile["relative_error"]

    print(summarize(tracer_frac, mass_frac, rel_err))
    plot_profiles(x, tracer_frac, mass_frac, rel_err, args.out)
    print(f"Wrote plot to {args.out}")


if __name__ == "__main__":
    main()
