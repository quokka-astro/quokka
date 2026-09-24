#!/usr/bin/env python3
"""Run the 20/40/60-cell rings and remake Krumholz et al. (2004), Fig. 10.

Requires NumPy and Matplotlib. Use --run once, then omit it to replot saved data.
All runs use one MPI rank and independent working directories. Existing run
files are never overwritten by --run. The paper curves come from eqs. 22/24,
not digitized Fig. 10 points. Quokka averages use actual sample times.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

RADII = (20, 40, 60)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cases(args):
    executable = args.executable.resolve()
    source_input = args.input.resolve()
    if not executable.is_file() or not source_input.is_file():
        raise ValueError("Executable and input must exist")
    for radius in RADII:
        directory = args.output / f"r{radius}"
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Refusing to overwrite {directory}; choose a fresh --output")
    input_copy = args.output / "input.toml"
    input_copy.write_bytes(source_input.read_bytes())
    binary_hash = sha256(executable)
    repo = Path(__file__).resolve().parents[3]
    provenance = {
        "executable": str(executable), "executable_sha256": binary_hash,
        "input_source": str(source_input), "input_sha256": sha256(input_copy),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in
                          sorted((repo / "src/problems/KeplerianDisk").glob("*")) if p.is_file()},
        "ranks_per_run": 1, "OMP_NUM_THREADS": "1", "jobs": args.jobs,
        "reconstruction_order": args.reconstruction_order, "density_floor": args.density_floor,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def one(radius):
        directory = args.output / f"r{radius}"
        directory.mkdir(exist_ok=True)
        command = [str(executable), str(input_copy), f"disk.ring_radius_cells={radius}",
                   "disk.orbits=0.9", "disk.measure_alpha=1", "disk.alpha_interval_orbits=0.01",
                   "disk.check_initial=1", "disk.check_alpha=1",
                   f"hydro.reconstruction_order={args.reconstruction_order}", f"density_floor={args.density_floor:.17g}"]
        meta = {"radius_cells": radius, "command": command, "cwd": str(directory),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (directory / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        start = time.monotonic()
        print(f"Starting R/dx={radius}", flush=True)
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1"}, check=False)
        meta.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
        (directory / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"R/dx={radius} failed; see {directory / 'run.log'}")
        if sha256(executable) != binary_hash:
            raise RuntimeError("Executable changed during the measurement campaign")
        print(f"Finished R/dx={radius} in {meta['elapsed_seconds']:.1f} s", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(one, RADII))


def analyze(path, radius, start=0.09, end=0.9):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 9 or len(data) < 2 or not np.all(np.diff(data[:, 1]) > 0):
        raise ValueError(f"Malformed or nonmonotonic alpha history: {path}")
    header = path.read_text().splitlines()[1]
    metadata = {key.strip(): float(value) for key, value in
                (item.split("=") for item in header.lstrip("# ").split(","))}
    if metadata["R0"] != radius:
        raise ValueError(f"Radius mismatch in {path}")
    area = 0.
    coverage = 0.
    valid = (data[:, 6] == 0) & np.isfinite(data[:, 4])
    for i in range(len(data) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        lo = max(start, data[i, 1])
        hi = min(end, data[i + 1, 1])
        if hi > lo:
            slope = (data[i + 1, 4] - data[i, 4]) / (data[i + 1, 1] - data[i, 1])
            area += (hi - lo) * (data[i, 4] + slope * ((lo + hi) / 2 - data[i, 1]))
            coverage += hi - lo
    if not np.isclose(coverage, end - start, rtol=0, atol=1e-10):
        raise ValueError(f"Incomplete averaging coverage for R/dx={radius}: {coverage}")
    mean = area / coverage
    if start == 0.09 and end == 0.9 and not np.isclose(mean, data[-1, 7], rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent average disagrees with in-situ result in {path}")
    window = (data[:, 1] >= start) & (data[:, 1] <= end + 1e-12) & valid
    residuals = np.r_[np.interp([start, end], data[:, 1], data[:, 5]), data[window, 5]]
    history = np.loadtxt(path.with_name("keplerian_disk_history.txt"), ndmin=2)
    row = {
        "radius_cells": radius, "alpha_mean": mean, "covered_orbits": coverage,
        "paper_ring_eq24": 1100 * radius**-2.37,
        "ratio_to_paper_ring_fit": mean / (1100 * radius**-2.37),
        "alpha_final": float(data[-1, 4]), "fit_relative_L2_final": float(data[-1, 5]),
        "fit_relative_L2_max_in_window": float(residuals.max()),
        "alpha_at_window_end": float(np.interp(end, data[:, 1], data[:, 4])),
        "samples_in_window": int(window.sum()), "samples": len(data), "invalid_samples": int((~valid).sum()),
        "mass_relative_change": float(history[-1, 1] / history[0, 1] - 1),
        "Lz_relative_change": float(history[-1, 2] / history[0, 2] - 1),
        "bondi_radius_cells": 1 / metadata["cs_squared"],
    }
    return data, row


def plot(args):
    provenance = json.loads((args.data_dir / "provenance.json").read_text())
    start, end = args.average_start, args.average_end
    interval_label = f"{start:g}–{end:g}"
    density_floor = provenance.get("density_floor", 1e-10)
    reconstruction = {3: "PPM", 5: "xPPM"}[provenance.get("reconstruction_order", 5)]
    histories, rows = zip(*(analyze(args.data_dir / f"r{r}/keplerian_disk_alpha.txt", r, start, end) for r in RADII))
    x = np.array(RADII, dtype=float)
    y = np.array([row["alpha_mean"] for row in rows])
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    normalization = np.exp(intercept)
    summary = {"reconstruction": reconstruction, "density_floor": density_floor, "plot_script_sha256": sha256(Path(__file__)),
               "data_directory": str(args.data_dir),
               "alpha_history_sha256": {str(r): sha256(args.data_dir / f"r{r}/keplerian_disk_alpha.txt") for r in RADII},
               "averaging_interval_orbits": [start, end], "measurements": rows,
               "quokka_power_law": {"normalization": normalization, "exponent": slope,
                                    "method": "unweighted least squares in log(alpha) vs log(R/dx), three points"},
               "paper_reference": {"eq22": "78 * 1.36e4 * (R/dx)^(-3.85)",
                                   "eq24": "1100 * (R/dx)^(-2.37)"}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "measurements.csv").open("w") as stream:
        stream.write(",".join(rows[0]) + "\n")
        for row in rows:
            stream.write(",".join(str(row[key]) for key in rows[0]) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    radii = np.geomspace(10, 70, 400)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    ax.loglog(radii, 78 * 1.36e4 * radii**-3.85, color="0.3", lw=1.8,
              label="Paper: disk evacuation (eq. 22)")
    ax.loglog(radii, 1100 * radii**-2.37, color="0.55", lw=1.8, ls="--",
              label="Paper: ring fit, 0.09–0.9 orbits (eq. 24)")
    ax.loglog(radii, normalization * radii**slope, color="#0072B2", lw=2.2,
              label=rf"Fit: $\alpha={normalization:.2f}(R/\Delta x)^{{{slope:.3f}}}$")
    ax.loglog(x, y, "o", color="#0072B2", ms=7, zorder=5, label="Quokka: measured ring averages")
    for radius, mean, offset, alignment in zip(x, y, [(7, 7), (-8, -18), (0, 14)], ["left", "right", "center"]):
        ax.annotate(f"{mean:.3g}", (radius, mean), xytext=offset, ha=alignment,
                    textcoords="offset points", color="#0072B2")
    ax.set(xlabel=r"Ring radius $R/\Delta x$", ylabel=r"Effective viscosity $\alpha$",
           title="Figure 10 comparison: ring spreading", xlim=(10, 70), ylim=(min(0.04, 0.7 * y.min()), max(15, 1.4 * y.max())))
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70], labels=["10", "20", "30", "40", "50", "60", "70"])
    ax.grid(which="major", color="0.9", lw=0.7)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.get_layout_engine().set(rect=(0, 0.095, 1, 0.905))
    fig.text(0.5, 0.025, f"Time averages: {interval_label} orbits • 2D, fixed central mass\n{reconstruction}, AV = 0.1, density floor = {density_floor:.0e}",
             ha="center", fontsize=9, color="0.35")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"figure10.{suffix}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.3), sharex=True, layout="constrained")
    for radius, data, row, color in zip(RADII, histories, rows, ["#0072B2", "#D55E00", "#009E73"]):
        valid = data[:, 6] == 0
        axes[0].plot(data[valid, 1], data[valid, 4], color=color, label=f"R/Δx = {radius}, mean = {row['alpha_mean']:.3g}")
        axes[1].plot(data[:, 1], data[:, 5], color=color)
    axes[0].set(yscale="log", ylabel=r"Fitted $\alpha$", title="Time history and profile-fit quality")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xlabel="Time / initial orbital period", ylabel="Relative profile-fit L2 residual", xlim=(0, .9))
    for ax in axes:
        ax.axvspan(0, start, color="0.93", zorder=-1)
        ax.axvspan(end, .9, color="0.93", zorder=-1)
        ax.grid(color="0.9", lw=.7)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"alpha_history.{suffix}", dpi=220)
    plt.close(fig)
    report = ["# Figure 10 comparison: Quokka ring spreading", "",
              f"Reconstruction: **{reconstruction}**. Density floor: **{density_floor:.0e}**. "
              "Artificial-viscosity coefficient: **0.1**. CFL: **0.3**.", "",
              "Each 2D 256² run uses a two-cell-wide ring, a fixed central point mass, "
              "10 K isothermal gas, and the same executable. Averages are time-weighted "
              f"over {interval_label} initial orbital periods; each has full {end-start:g}-orbit coverage.", "",
              "| R/Δx | Mean alpha | Final alpha | Final relative fit L2 | Relative mass change |",
              "|---:|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['radius_cells']} | {row['alpha_mean']:.8g} | {row['alpha_final']:.8g} | "
                      f"{row['fit_relative_L2_final']:.6g} | {row['mass_relative_change']:.3g} |")
    report += ["", f"Descriptive three-point log-space fit: alpha = {normalization:.6g} (R/Δx)^({slope:.6g}).", "",
               "[Figure 10 PNG](figure10.png) · [PDF](figure10.pdf) · [Time histories](alpha_history.png) · "
               "[Measurements CSV](measurements.csv)", "",
               "The paper reference lines use Krumholz, McKee & Klein (2004), ApJ 611, 399, "
               "section 3.4.2: equation (22), 78 × 1.36e4 × (R/Δx)^(-3.85), and equation (24), "
               "1100 × (R/Δx)^(-2.37). The latter is the published ring fit, not digitized measurements. "
               "The Quokka Bondi radius is " + f"{rows[0]['bondi_radius_cells']:.6g} cells.", "",
               "Alpha is inferred from pressureless constant-viscosity profile fits. Finite initial width, "
               "pressure-driven evolution, and numerical artifacts can affect it. Inspect fit residuals "
               "and the time-history figure; the three-point power law is descriptive, with no "
               "statistical error bars inferred from these deterministic runs.", "",
               f"Source run directory: `{args.data_dir}`. Exact commands, exit codes, timings, "
               "executable/input/source hashes, and raw diagnostic histories are retained there. "
               "The current summary records the alpha-history hashes used in this analysis.", ""]
    (args.output / "README.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run all three cases before plotting")
    parser.add_argument("--executable", type=Path, default=Path("build/2d/src/problems/KeplerianDisk/KeplerianDisk"))
    parser.add_argument("--input", type=Path, default=Path("inputs/KeplerianDisk.toml"))
    parser.add_argument("--output", type=Path, default=Path("build/2d/keplerian-figure10"))
    parser.add_argument("--data-dir", type=Path, help="Saved campaign directory; defaults to --output")
    parser.add_argument("--average-start", type=float, default=0.09)
    parser.add_argument("--average-end", type=float, default=0.9)
    parser.add_argument("--density-floor", type=float, default=1e-10)
    parser.add_argument("--reconstruction-order", type=int, choices=(3, 5), default=3)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if not np.isfinite(args.density_floor) or args.density_floor <= 0:
        parser.error("--density-floor must be positive and finite")
    args.output = args.output.resolve()
    args.data_dir = args.data_dir.resolve() if args.data_dir else args.output
    if not (np.isfinite(args.average_start) and np.isfinite(args.average_end) and
            0 <= args.average_start < args.average_end <= 0.9):
        parser.error("Require 0 <= --average-start < --average-end <= 0.9")
    if args.run and args.data_dir != args.output:
        parser.error("--run requires --data-dir to match --output")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run:
        run_cases(args)
    plot(args)


if __name__ == "__main__":
    main()
