#!/usr/bin/env python3
"""Plot the resistive-MHD energy-balance diagnostic profile."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def read_csv(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    columns = {name: [] for name in rows[0].keys()}
    for row in rows:
        for name, value in row.items():
            columns[name].append(float(value))
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("resistive_mhd_energy_balance.csv"),
        help="CSV produced by the ResistiveMHDEnergyBalance problem.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resistive_mhd_energy_balance.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quokka-matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(data["x"], data["heating_rate"], color="black", linewidth=1.8, label="Quokka")
    ax.plot(data["x"], data["heating_rate_correct"], color="#1b9e77", linestyle="--", linewidth=1.5, label="correct eta J^2")
    ax.plot(
        data["x"],
        data["heating_rate_missing_flux"],
        color="#d95f02",
        linestyle=":",
        linewidth=1.8,
        label="missing-flux prediction",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("(p - p_initial) / t")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
