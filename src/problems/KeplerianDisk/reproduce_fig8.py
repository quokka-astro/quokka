#!/usr/bin/env python3
"""Apply the Fig. 8 density threshold to saved, single-level 2-D disk runs.

Requires NumPy and Matplotlib. The narrow AMReX reader deliberately rejects
other layouts. Radius uses linear interpolation of the 90% threshold between
one-cell annular samples, with no background subtraction. The paper does not specify threshold interpolation. Use --power-law-run for
the continuous disk; the default analyzes the earlier three ring runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np

BOX = re.compile(r"\(\((-?\d+),(-?\d+)\) \((-?\d+),(-?\d+)\) \(0,0\)\)")


def read_density(path):
    lines = (path / "Header").read_text().splitlines()
    ncomp = int(lines[1])
    fields = lines[2:2 + ncomp]
    h = lines[2 + ncomp:]
    if int(h[0]) != 2 or int(h[2]) != 0:
        raise ValueError(f"Expected single-level 2D plotfile: {path}")
    time = float(h[1])
    lo = np.fromstring(h[3], sep=" ")
    hi = np.fromstring(h[4], sep=" ")
    domain = BOX.fullmatch(h[6].strip())
    if not domain:
        raise ValueError(f"Unsupported domain: {path}")
    x0, y0, x1, y1 = map(int, domain.groups())
    shape = (x1 - x0 + 1, y1 - y0 + 1)
    dx = (hi - lo) / shape
    density = np.full(shape, np.nan)
    occupied = np.zeros(shape, dtype=bool)
    cell = (path / "Level_0/Cell_H").read_text().splitlines()
    if cell[:4] != ["1", "1", str(ncomp), "0"]:
        raise ValueError(f"Unsupported FAB layout: {path}")
    for line in cell:
        if not line.startswith("FabOnDisk:"):
            continue
        _, filename, offset = line.split()
        with (path / "Level_0" / filename).open("rb") as f:
            f.seek(int(offset))
            header = f.readline().decode("ascii")
            if not header.startswith("FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"):
                raise ValueError(f"Expected little-endian IEEE float64 FAB: {path}")
            box = BOX.search(header)
            a, b, c, d = map(int, box.groups())
            nx, ny = c - a + 1, d - b + 1
            f.seek(fields.index("gasDensity") * nx * ny * 8, 1)
            values = np.fromfile(f, dtype="<f8", count=nx * ny)
            region = np.s_[a-x0:c-x0+1, b-y0:d-y0+1]
            if occupied[region].any() or values.size != nx * ny:
                raise ValueError(f"Overlapping or truncated FAB: {path}")
            density[region] = values.reshape((nx, ny), order="F")
            occupied[region] = True
    if not occupied.all() or not np.isfinite(density).all():
        raise ValueError(f"Incomplete density: {path}")
    return time, density, lo, dx


def edge(profile, initial, radius):
    """First outward zero of Sigma(t)-0.9 Sigma(0), outside the excluded core.

    No extrapolation through the excluded region. An already-positive first
    eligible sample gives no resolved interface and returns NaN.
    """
    keep = radius >= 4.
    r = np.asarray(radius)[keep]
    f = (np.asarray(profile) - 0.9 * np.asarray(initial))[keep]
    if not len(r) or not np.all(np.isfinite(f)) or np.any(np.asarray(initial)[keep] <= 0):
        return float("nan")
    if f[0] > 0:
        return float("nan")
    if f[0] == 0:
        return float(r[0])
    above = np.flatnonzero(f >= 0)
    if not above.size:
        return float("nan")
    j = above[0]
    return float(r[j-1] - f[j-1] * (r[j] - r[j-1]) / (f[j] - f[j-1]))


def analyze(directory, output, ring):
    paths = sorted(directory.glob("plt[0-9]*"))
    time, initial, lo, dx = read_density(paths[0])
    if time != 0 or not np.allclose(dx, 1) or not np.allclose(lo, -128) or initial.shape != (256, 256):
        raise ValueError("Expected existing centered 256-square, unit-cell ring runs")
    x = lo[0] + (np.arange(initial.shape[0]) + 0.5) * dx[0]
    y = lo[1] + (np.arange(initial.shape[1]) + 0.5) * dx[1]
    r = np.hypot(x[:, None], y[None, :])
    bins = np.floor(r).astype(int)
    mask = bins < 128
    counts = np.bincount(bins[mask], minlength=128)
    radius = np.arange(128) + 0.5
    def profile(density):
        return np.bincount(bins[mask], weights=density[mask], minlength=128) / counts
    initial_profile = profile(initial)
    if ring is None:
        outer_radius = 2.e15 / (1.85 * 1.495978707e13)
        expected = np.where(r <= outer_radius, outer_radius / r, 1.e-6)
    else:
        expected = np.where(np.abs(r - ring) < 1, 1., 1.e-6)
    np.testing.assert_allclose(initial, expected, rtol=1.e-14, atol=0)
    history = np.loadtxt(directory / "keplerian_disk_history.txt")
    rows, profiles = [], []
    for path in paths:
        t, density, current_lo, current_dx = read_density(path)
        np.testing.assert_array_equal(current_lo, lo)
        np.testing.assert_array_equal(current_dx, dx)
        step = int(path.name[3:])
        np.testing.assert_allclose(t, history[step, 0], rtol=1.e-12, atol=1.e-12)
        np.testing.assert_allclose(density.sum() * dx.prod(), history[step, 1], rtol=1.e-10)
        sigma = profile(density)
        revac = edge(sigma, initial_profile, radius)
        rows.append((step, t, t / (2*np.pi*4**1.5), (t / (2*np.pi*ring**1.5) if ring is not None else float("nan")), revac, 1.85*revac, int(sigma[4] > 0.9 * initial_profile[4])))
        profiles.append(sigma)
    data = np.asarray(rows)
    if not np.all(np.diff(data[:, 1]) > 0):
        raise ValueError("Nonmonotonic plotfile times")
    name = f"r{ring}" if ring is not None else "power_law"
    np.savetxt(output / f"{name}.csv", data, delimiter=",", comments="",
               header="step,time_code,time_periods_at_4dx,time_ring_orbits,r_evac_cells,r_evac_AU,inner_edge_unresolved")
    np.savez_compressed(output / f"{name}_profiles.npz", radius_cells=radius,
                        initial_surface_density=initial_profile, surface_density=np.asarray(profiles),
                        time_code=data[:, 1])
    meta = {"ring_radius_cells": ring, "run_directory": str(directory.resolve()),
            "plotfiles": len(paths), "run_json_sha256": hashlib.sha256((directory / "run.json").read_bytes()).hexdigest(),
            "final_time_ring_orbits": float(data[-1, 3]) if ring is not None else None, "final_time_periods_at_4dx": data[-1, 2],
            "final_radius_cells": data[-1, 4], "final_radius_AU": data[-1, 5],
            "min_radius_cells": float(np.nanmin(data[:, 4])), "max_radius_cells": float(np.nanmax(data[:, 4])),
            "missing_crossings": int(np.isnan(data[:, 4]).sum()),
            "inner_edge_unresolved": int(data[:, 6].sum())}
    print(json.dumps(meta), flush=True)
    return data, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-law-run", type=Path, help="Analyze one continuous-disk run instead of the three rings")
    parser.add_argument("--build-dir", type=Path, default=Path("build/2d"))
    parser.add_argument("--output", type=Path, default=Path("build/2d/keplerian-figure8-ppm"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Manufactured linear threshold with a nonconstant initial density:
    # interpolate the density difference, not the ratio or rounded bin index.
    rr = np.arange(12) + .5
    initial = 10. / rr
    manufactured = .9 * initial + .02 * (rr - 7.23)
    np.testing.assert_allclose(edge(manufactured, initial, rr), 7.23, atol=1e-14)
    np.testing.assert_allclose(edge(.9 * initial + .02 * (rr - 7.5), initial, rr), 7.5)
    assert np.isnan(edge(np.ones(12), np.ones(12), rr))
    assert np.isnan(edge(np.zeros(12), np.ones(12), rr))
    # Select the first outward crossing even if another depleted region follows.
    f = np.array([-1., -1., -1., -1., -.2, .6, -.5, 1., 1., 1., 1., 1.])
    np.testing.assert_allclose(edge(.9 + f, np.ones(12), rr), 4.75)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    records = []
    cases = [(None, "D")] if args.power_law_run else list(zip((20, 40, 60), ("D", "+", "*")))
    for ring, marker in cases:
        directory = args.power_law_run if ring is None else args.build_dir / f"keplerian-r{ring}-ppm-plots10-20260910"
        data, meta = analyze(directory, args.output, ring)
        records.append(meta)
        positive = data[:, 2] > 0
        ax.loglog(data[positive, 2], data[positive, 5], color="black", lw=.8,
                  marker=marker, markersize=4, markerfacecolor="none", markeredgewidth=.6,
                  markevery=max(1, int(positive.sum()/14)),
                  label="Power-law disk, no accretion" if ring is None else rf"$R_0/\Delta x={ring}$")
    ax.set_xlabel("Time (Periods)")
    ax.set_ylabel("Radius (AU)")
    secondary = ax.secondary_yaxis("right", functions=(lambda au: au / 1.85, lambda cells: cells * 1.85))
    secondary.set_ylabel("Radius (cells)")
    for axis in (ax.xaxis, ax.yaxis, secondary.yaxis):
        axis.set_major_formatter(ScalarFormatter())
    if args.power_law_run:
        ax.yaxis.set_major_locator(FixedLocator([5, 8, 10, 15, 20, 30, 50, 100, 200]))
        secondary.yaxis.set_major_locator(FixedLocator([4, 5, 6, 8, 10, 20, 40, 80, 100]))
    else:
        ax.yaxis.set_major_locator(FixedLocator([8, 10, 12, 14, 16]))
        secondary.yaxis.set_major_locator(FixedLocator([4, 5, 6, 7, 8]))
    ax.xaxis.set_major_locator(FixedLocator([.01, .1, 1, 5, 10, 20, 50, 100]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    secondary.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True)
    secondary.tick_params(which="both", direction="in")
    ax.legend(frameon=False, fontsize=9)
    fig.text(.5, .015, r"Periods at $r=4\Delta x$; $\Delta x=1.85$ AU; 90% initial density", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"figure8.{suffix}", dpi=200)
    plt.close(fig)
    summary = {"threshold": .9, "excluded_radius_cells": 4, "annular_bin_width_cells": 1,
               "edge_convention": "First outward zero of linearly interpolated Sigma(t)-0.9 Sigma(0); NaN if unbracketed",
               "background_subtracted": False, "cell_size_AU": 1.85,
               "time_unit": "Keplerian period at 4 dx, GM=1: 16 pi code time units",
               "runs": records}
    summary["profile"] = "power_law" if args.power_law_run else "ring"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.power_law_run:
        (args.output / "README.md").write_text("""# Continuous-disk Figure 8 diagnostic

Section 3.4.1 initial density: Sigma = 0.1 (r0/r) g/cm^2 inside r0=2e15 cm.
Fixed solar-mass potential, 10 K isothermal gas, no accretion or self-gravity.
Retains dx=1.85 AU from the existing runs, PPM order 3 and floor 1e-7 Sigma_0.
Time is in orbital periods at 4 dx; this run spans 50 such periods.

One-cell annular averages are compared to the actual initial plotfile. The
first outward zero of F(r)=Sigma(t)-0.9 Sigma(0) outside 4 dx is the edge.
For adjacent samples F_i<0<=F_j, use r=r_i-F_i*(r_j-r_i)/(F_j-F_i).
No smoothing or background subtraction is applied. If the first eligible
sample (4.5 dx) already meets the threshold, the interface is unresolved
inside the cutoff; CSV records NaN and inner_edge_unresolved=1. Such times
are omitted from the log plot. This avoids inventing a boundary position or
interpolating through the excluded hydrostatic core. Sub-cell precision is
an interpolation estimate, not a demonstrated sub-cell error bound.
The central four cells are excluded following the paper's no-accretion case.
Logarithmic axes, AU/cell scales, and diamond symbols follow Figure 8, with
axis limits adapted to these data. The paper's accreting/advected cases are
not simulated here. The adopted cell length differs from its printed cm value.

Every plotfile time and integrated density is checked against in-situ history;
the initial density is checked cell-by-cell against the power-law prescription.
CSV contains all saved times (the ring-orbits column is inapplicable, NaN).
NPZ stores all annular profiles and the initial reference; summary.json records
run metadata. Density profiles are in units of 0.1 g/cm^2.
""")
        return
    (args.output / "README.md").write_text('''# Figure 8 diagnostic on the three ring runs

The plot follows Figure 8's log-log axes, paired AU/cell radius axes, and
black diamond/plus/star curves. Markers identify ring radii, not the original
paper's accretion treatments. Axis limits adapt to these data. Time is measured
in orbital periods at 4 dx, the reference radius of equation (18). All positive
times are plotted; t=0 is retained in CSV but cannot appear on a logarithmic axis.

At each plotfile, gasDensity is averaged over full one-cell annuli centered on
the origin using equal cell-area weights. In these 2D runs it represents surface
density. The actual initial plotfile provides the comparison profile. The edge
is the first outward zero of linearly interpolated Sigma(t)-0.9 Sigma(0).
We exclude the inner four cells as prescribed for the paper's no-accretion run.
No background subtraction, ring mask, or smoothing is applied.
The radial sampling is one cell; centers are 4.5, 5.5, ... dx. If the first
eligible sample meets the threshold already, the inner edge is unresolved
and is recorded as NaN with inner_edge_unresolved=1. Missing crossings
would be recorded as NaN, with gaps in the plot. Profiles extend to 128 dx.

These are narrow-ring tests from section 3.4.2, not the continuous Sigma~1/r
disks of Figure 8 (section 3.4.1). Initially the entire interior of each ring
contains background gas at 1e-6. The threshold can select that gas and should
not be interpreted as the geometrical ring edge or a recreation of the paper's
continuous-disk evacuation experiment. The simulation's adopted dx=1.85 AU is
used consistently for the AU axis (the paper's printed length conversion is
inconsistent). Each run uses PPM order 3 and density floor 1e-7.

CSV files contain every saved output (10-step cadence plus endpoints). NPZ files
contain all radial profiles and their initial references. Every plotfile time
and integrated density was checked against the independent in-situ history;
initial densities were checked against the exact top-hat initial condition.

Reproduce from the repository root:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/reproduce_fig8.py
```
''')


if __name__ == "__main__":
    main()
