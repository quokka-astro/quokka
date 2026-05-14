#!/usr/bin/env python3
"""Analyze SN feedback over timesteps and plot gas density with SN markers.

This script is designed for ChuhanProblem outputs.
It checks each plotfile for SN feedback evidence and produces one density
slice image per timestep. If SN positions are available, they are marked.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import yt


PLOTFILE_REGEX = re.compile(r"^plt\d{7}$")


def discover_plotfiles(patterns: list[str]) -> list[str]:
    """Expand patterns/directories and return sorted plotfile paths."""
    found: list[str] = []
    for item in patterns:
        matches = sorted(glob.glob(item))
        if matches:
            found.extend(matches)
            continue
        if os.path.isdir(item):
            found.append(item)

    # Keep only canonical AMReX plotfiles like plt0000000.
    filtered = []
    for path in found:
        name = os.path.basename(path.rstrip("/"))
        if PLOTFILE_REGEX.match(name):
            filtered.append(path)

    # Remove duplicates while preserving sort order.
    return sorted(set(filtered))


def has_particle_payload(plotfile: str, ptype_dir: str = "StochasticStellarPop_particles") -> bool:
    """Return True if particle payload files exist beyond Header/Fields.yaml."""
    pdir = Path(plotfile) / ptype_dir
    if not pdir.exists():
        return False

    files = [p for p in pdir.rglob("*") if p.is_file()]
    payload = [p for p in files if p.name not in {"Header", "Fields.yaml"}]
    return len(payload) > 0


def extract_sn_positions(ds: yt.Dataset) -> tuple[list[tuple[float, float]], int]:
    """Extract SN marker positions in code_length units if available.

    Returns (positions, n_particles_seen).
    positions are (x, y) for z-slice markers.
    """
    ad = ds.all_data()

    # Most current outputs have no particle payload for this test case.
    ptypes = [pt for pt in ds.particle_types if pt != "all"]
    if len(ptypes) == 0:
        return [], 0

    # Try to infer SN from particle stage/death position if fields are available.
    # Stage enum: SNRemnant == 2.
    stage_names = ["evolution_stage"]
    death_x_names = ["death_x", "particle_position_x"]
    death_y_names = ["death_y", "particle_position_y"]

    positions: list[tuple[float, float]] = []
    n_particles_total = 0

    for ptype in ptypes:
        # Probe available fields for this ptype.
        keys = set()
        try:
            for key in ds.field_list:
                if isinstance(key, tuple) and len(key) == 2 and key[0] == ptype:
                    keys.add(key[1])
        except Exception:
            pass

        if len(keys) == 0:
            continue

        stage_field = next((f for f in stage_names if f in keys), None)
        x_field = next((f for f in death_x_names if f in keys), None)
        y_field = next((f for f in death_y_names if f in keys), None)

        # Need at least stage and coordinates to place markers.
        if stage_field is None or x_field is None or y_field is None:
            continue

        try:
            stage = ad[(ptype, stage_field)]
            xvals = ad[(ptype, x_field)].to("code_length")
            yvals = ad[(ptype, y_field)].to("code_length")
            n_particles_total += int(stage.size)
            for i in range(int(stage.size)):
                if int(stage[i]) == 2:
                    positions.append((float(xvals[i]), float(yvals[i])))
        except Exception:
            continue

    return positions, n_particles_total


def scalar_mass_sums(ds: yt.Dataset) -> dict[str, float]:
    """Compute domain-integrated scalar sums for quick SN-source sanity checks."""
    ad = ds.all_data()
    out: dict[str, float] = {}
    for name in ["scalar_0", "scalar_1", "scalar_2"]:
        try:
            out[name] = float(ad[("boxlib", name)].sum())
        except Exception:
            out[name] = float("nan")
    return out


def render_density(plotfile: str, outdir: Path, sn_positions_xy: list[tuple[float, float]]) -> str:
    """Render gas density z-slice and mark SN positions if present."""
    ds = yt.load(plotfile)
    ts = os.path.basename(plotfile).replace("plt", "")

    slc = yt.SlicePlot(ds, "z", ("boxlib", "gasDensity"), center="c", width=(1.0, "code_length"))
    slc.set_cmap(("boxlib", "gasDensity"), "magma")
    slc.annotate_grids(linewidth=0.4, alpha=0.25)

    if sn_positions_xy:
        for x, y in sn_positions_xy:
            slc.annotate_marker((x, y), coord_system="data", plot_args={"color": "cyan", "s": 40, "marker": "x"})
        slc.annotate_text((0.03, 0.03), f"SN markers: {len(sn_positions_xy)}", coord_system="axis", text_args={"color": "white"})
    else:
        slc.annotate_text((0.03, 0.03), "No SN detected", coord_system="axis", text_args={"color": "white"})

    outbase = outdir / f"density_ts{ts}"
    slc.save(str(outbase), mpl_kwargs={"dpi": 150, "bbox_inches": "tight"})
    return f"{outbase}_Slice_z_boxlib_gasDensity.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SN feedback and plot density with SN markers.")
    parser.add_argument("plotdirs", nargs="+", help="Plotfiles or glob patterns, e.g. tests/plt000000?")
    parser.add_argument("-o", "--output", default="tests/chuhan_density_sn_plots", help="Output directory for density images")
    parser.add_argument("--summary-csv", default="tests/chuhan_density_sn_plots/sn_feedback_summary.csv", help="CSV summary path")
    args = parser.parse_args()

    yt.set_log_level(40)

    plotfiles = discover_plotfiles(args.plotdirs)
    if not plotfiles:
        raise SystemExit("No canonical plotfiles found. Use names like plt0000000.")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.summary_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    any_sn = False

    for pf in plotfiles:
        ds = yt.load(pf)
        time_myr = float(ds.current_time.to("Myr"))
        has_payload = has_particle_payload(pf)
        sn_positions, n_particles = extract_sn_positions(ds)
        sums = scalar_mass_sums(ds)
        png = render_density(pf, outdir, sn_positions)

        n_sn = len(sn_positions)
        any_sn = any_sn or (n_sn > 0)

        rows.append(
            {
                "plotfile": pf,
                "time_myr": time_myr,
                "has_particle_payload": int(has_payload),
                "n_particles_seen": n_particles,
                "n_sn_markers": n_sn,
                "scalar_0_sum": sums["scalar_0"],
                "scalar_1_sum": sums["scalar_1"],
                "scalar_2_sum": sums["scalar_2"],
                "density_png": png,
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plotfile",
                "time_myr",
                "has_particle_payload",
                "n_particles_seen",
                "n_sn_markers",
                "scalar_0_sum",
                "scalar_1_sum",
                "scalar_2_sum",
                "density_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analyzed {len(rows)} plotfiles.")
    print(f"SN detected across run: {'YES' if any_sn else 'NO'}")
    print(f"Summary CSV: {csv_path}")
    print(f"Density images: {outdir}")


if __name__ == "__main__":
    main()