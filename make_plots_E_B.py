#!/usr/bin/env python3
"""
plot_magnetic_energy.py

Reads a Quokka ComputeStatistics() diagnostics file (the "stats" output produced
by testMHDDisk.cpp) and plots the toroidal, poloidal, and total magnetic energy
within the annulus (R in [2, 8] kpc, |z| < 0.5 kpc) as a function of time.

Handles:
  - the "# cycle time ..." header line (used to name columns)
  - "## Simulation restarted at: ..." comment lines from checkpoint-chained runs
    (can appear multiple times if the job was restarted several times)
  - duplicate cycle numbers across restarts (keeps the last occurrence, since a
    restart re-writes statistics for the cycle it resumed from)

Usage:
    python plot_magnetic_energy.py <stats_file> [-o output.png]
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

SECONDS_PER_YEAR = 3.15576e7  # matches testMHDDisk.cpp's seconds_per_year constant


def parse_stats_file(path):
    """Parse a Quokka stats file into a dict of column_name -> np.ndarray.

    Restart markers ("##...") are skipped. The most recent "# ..." header line
    found defines column names (in case a restarted run's header differs).
    Duplicate cycle numbers (from restarts re-emitting the same cycle) are
    resolved by keeping the last occurrence in file order.
    """
    columns = None
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("##"):
                # restart marker, e.g. "## Simulation restarted at: ..."
                continue
            if line.startswith("#"):
                # header line, e.g. "# cycle time disk_mass divB_max ..."
                columns = line.lstrip("#").split()
                continue
            # data row
            values = line.split()
            if columns is None:
                raise ValueError(
                    f"Encountered a data row before any header line was found: {line!r}"
                )
            if len(values) != len(columns):
                # tolerate malformed/truncated trailing rows (e.g. a run that
                # was killed mid-write) rather than crashing the whole parse
                print(
                    f"Warning: skipping malformed row (expected {len(columns)} "
                    f"fields, got {len(values)}): {line!r}",
                    file=sys.stderr,
                )
                continue
            rows.append([float(v) for v in values])

    if columns is None or not rows:
        raise ValueError(f"No parseable data found in {path}")

    data = np.array(rows)
    col_index = {name: i for i, name in enumerate(columns)}

    # Resolve duplicate cycle numbers (from restarts) by keeping the last
    # occurrence for each cycle, then re-sort by cycle to guarantee monotonic order.
    cycle_col = data[:, col_index["cycle"]]
    last_index_for_cycle = {}
    for row_idx, cyc in enumerate(cycle_col):
        last_index_for_cycle[cyc] = row_idx  # later occurrences overwrite earlier ones
    keep_indices = sorted(last_index_for_cycle.values(), key=lambda i: cycle_col[i])
    data = data[keep_indices]

    return {name: data[:, i] for name, i in col_index.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_file", help="Path to the Quokka stats text file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (default: <stats_file stem>_energy.png)"
    )
    args = parser.parse_args()

    stats = parse_stats_file(args.stats_file)

    required = ["time", "energy_Btor_annulus", "energy_Bpol_annulus", "energy_Btot_annulus"]
    missing = [c for c in required if c not in stats]
    if missing:
        raise KeyError(f"Stats file is missing expected column(s): {missing}")

    time_myr = stats["time"] / SECONDS_PER_YEAR / 1.0e6

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(time_myr, stats["energy_Btor_annulus"], label=r"$E_{\rm tor}$ (toroidal)",
            color="#d62728", lw=2)
    ax.plot(time_myr, stats["energy_Bpol_annulus"], label=r"$E_{\rm pol}$ (poloidal)",
            color="#1f77b4", lw=2)
    ax.plot(time_myr, stats["energy_Btot_annulus"], label=r"$E_{\rm tot}$ (total)",
            color="0.3", lw=1.5, ls="--")

    ax.set_yscale("log")
    ax.set_xlabel("Time [Myr]")
    ax.set_ylabel("Magnetic energy [erg]")
    ax.set_title(r"Magnetic energy vs. time (annulus: $R \in [2, 8]$ kpc, $|z| < 0.5$ kpc)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    output_path = args.output
    if output_path is None:
        import os
        stem = os.path.splitext(os.path.basename(args.stats_file))[0]
        output_path = f"{stem}_energy.png"

    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    print(f"Parsed {len(stats['time'])} unique cycles "
          f"(t = 0 to {time_myr[-1]:.2f} Myr)")


if __name__ == "__main__":
    main()