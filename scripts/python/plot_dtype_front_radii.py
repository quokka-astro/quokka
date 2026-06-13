#!/usr/bin/env python3
"""Plot DTypeFront radius history from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM_PER_PC = 3.0856775814913673e18
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DTypeFront effective and analytical radii from a CSV file.")
    parser.add_argument("csv_file", type=Path, help="Input CSV with time, r_effective, and r_analytical columns.")
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix. Defaults to '<csv stem>_from_csv' beside the input file.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats passed to matplotlib savefig. Defaults to 'png,pdf'.",
    )
    parser.add_argument("--title", default="DTypeFront ionization-front radius", help="Figure title.")
    return parser.parse_args()


def read_radius_history(csv_file: Path) -> tuple[list[float], list[float], list[float]]:
    required_columns = ("time", "r_effective", "r_analytical")
    time_kyr: list[float] = []
    r_effective_pc: list[float] = []
    r_analytical_pc: list[float] = []

    with csv_file.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in required_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_file} is missing required column(s): {', '.join(missing)}")

        for row in reader:
            time_kyr.append(float(row["time"]) / SECONDS_PER_YEAR / 1.0e3)
            r_effective_pc.append(float(row["r_effective"]) / CM_PER_PC)
            r_analytical_pc.append(float(row["r_analytical"]) / CM_PER_PC)

    if not time_kyr:
        raise ValueError(f"No data rows found in {csv_file}")

    return time_kyr, r_effective_pc, r_analytical_pc


def make_plot(
    time_kyr: Sequence[float],
    r_effective_pc: Sequence[float],
    r_analytical_pc: Sequence[float],
    title: str,
) -> plt.Figure:
    residual_percent = [
        (effective - analytical) / analytical * 100.0 for effective, analytical in zip(r_effective_pc, r_analytical_pc)
    ]

    fig, (ax_radius, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_radius.plot(time_kyr, r_effective_pc, label="effective", color="C0", linewidth=2.0)
    ax_radius.plot(time_kyr, r_analytical_pc, label="analytical", color="black", linestyle="--", linewidth=1.8)
    ax_radius.set_ylabel("radius (pc)")
    ax_radius.legend(frameon=False)
    ax_radius.grid(True, alpha=0.25)

    ax_residual.axhline(0.0, color="0.35", linewidth=0.9)
    ax_residual.plot(time_kyr, residual_percent, color="C1", linewidth=1.8)
    ax_residual.set_xlabel("time (kyr)")
    ax_residual.set_ylabel("diff (%)")
    ax_residual.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    time_kyr, r_effective_pc, r_analytical_pc = read_radius_history(args.csv_file)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.csv_file.with_name(f"{args.csv_file.stem}_from_csv")

    figure = make_plot(time_kyr, r_effective_pc, r_analytical_pc, args.title)
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("At least one output format is required.")

    output_paths = []
    for fmt in formats:
        output_path = output_prefix.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=180)
        output_paths.append(output_path)
    plt.close(figure)

    residual_percent = [
        (effective - analytical) / analytical * 100.0 for effective, analytical in zip(r_effective_pc, r_analytical_pc)
    ]
    print(f"wrote {', '.join(str(path) for path in output_paths)} ({len(time_kyr)} rows)")
    print(f"time range: {min(time_kyr):.3g}-{max(time_kyr):.3g} kyr")
    print(f"effective radius range: {min(r_effective_pc):.3g}-{max(r_effective_pc):.3g} pc")
    print(f"final relative difference: {residual_percent[-1]:.3g}%")


if __name__ == "__main__":
    main()
