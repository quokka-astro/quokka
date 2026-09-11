#!/usr/bin/env python3
"""Plot one or more final KeplerianDisk radial transport profiles."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", help="One legend label per profile")
    parser.add_argument("--output", type=Path, default=Path("keplerian_disk_transport.png"))
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.profiles):
        parser.error("--labels must have one label per profile")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for index, path in enumerate(args.profiles):
        data = np.loadtxt(path, ndmin=2)
        if data.shape[1] != 6 or not np.isfinite(data).all():
            parser.error(f"{path}: expected six finite columns")
        time = next((line.split("=", 1)[1].strip() for line in path.read_text().splitlines() if line.startswith("# time =")), "unknown")
        label = args.labels[index] if args.labels else path.stem
        label += f" (t={float(time):.4g})" if time != "unknown" else ""
        axes[0].plot(data[:, 0], data[:, 3], label=label)
        axes[1].plot(data[:, 0], data[:, 4], label=label)
    axes[0].set_title("Instantaneous advective transport — positive outward")
    axes[0].set_ylabel(r"Mass flux $\dot M_{\rm out}$")
    axes[1].set_ylabel(r"Angular momentum flux $\dot L_{z,\rm out}$")
    axes[1].set_xlabel("Radius [code units]")
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.axhline(0, color="0.3", linewidth=0.7)
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
