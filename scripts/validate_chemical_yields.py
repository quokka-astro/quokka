#!/usr/bin/env python3
"""Validate gas scalar masses against tabulated chemical yields."""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import yt

MSUN_CGS = 1.9884e33
WR_AGB_WINDOW = 1.0e20

CHANNEL_TABLES = {
    "SNII": "SNII_yield_table.csv",
    "WR": "WR_yield_table.csv",
    "AGB": "AGB_yield_table.csv",
}


def normalize_numeric(token):
    return token.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", "")


def load_datatable(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if int(lines[0]) != 1:
        raise SystemExit(f"{path} is not a 1D DataTable")
    n_x = int(lines[1])
    n_out = int(lines[2])
    input_name = lines[3]
    output_names = [name.strip().lower() for name in lines[4].split(",")]
    input_unit = lines[5]
    xlo = float(normalize_numeric(lines[7]))
    xhi = float(normalize_numeric(lines[8]))
    spacing = lines[9].lower()
    if input_name != "mass" or input_unit != "Msun":
        raise SystemExit(f"{path} must use mass [Msun] as its coordinate")
    if len(output_names) != n_out:
        raise SystemExit(f"{path} output metadata length does not match Nout")
    values = {}
    for name, row in zip(output_names, lines[10 : 10 + n_out]):
        data = [float(normalize_numeric(token)) for token in row.split(",")]
        if len(data) != n_x:
            raise SystemExit(f"{path} row for {name} has {len(data)} values, expected {n_x}")
        values[name] = np.array(data)
    return {"n_x": n_x, "xlo": xlo, "xhi": xhi, "spacing": spacing, "values": values}


def load_yield_tables(root):
    root = Path(root)
    return {channel: load_datatable(root / filename) for channel, filename in CHANNEL_TABLES.items()}


def coordinate(table, mass_msun):
    mass_msun = max(mass_msun, 1.0e-12)
    if table["spacing"] in ("log", "fast_log"):
        return math.log(mass_msun)
    return mass_msun


def query_fraction(tables, channel, isotope, mass_msun):
    table = tables[channel]
    values = table["values"].get(isotope.lower())
    if values is None:
        raise SystemExit(f"isotope {isotope} not found in {channel} yield table")
    xlo = coordinate(table, table["xlo"])
    xhi = coordinate(table, table["xhi"])
    x = min(max(coordinate(table, mass_msun), xlo), xhi)
    position = (x - xlo) / ((xhi - xlo) / (table["n_x"] - 1))
    nearest = round(position)
    if 0 <= nearest < table["n_x"] and abs(position - nearest) < 1.0e-10:
        return float(values[nearest])
    lower = min(max(math.floor(position), 0), table["n_x"] - 2)
    frac = min(max(position - lower, 0.0), 1.0)
    return float((1.0 - frac) * values[lower] + frac * values[lower + 1])


def latest_plotfile(plotdir):
    plotfiles = sorted(path for path in glob.glob(str(Path(plotdir) / "plt*")) if ".old." not in Path(path).name)
    if not plotfiles:
        raise SystemExit(f"No plotfiles found in {plotdir}")
    return plotfiles[-1]


def scalar_masses(ds, start=0, count=3):
    data = ds.all_data()
    cell_volume = data[("index", "cell_volume")].to_value("cm**3")
    return [float((data[("boxlib", f"scalar_{start + i}")].to_value() * cell_volume).sum()) for i in range(count)]


def validate_snii(args):
    isotopes = ["C12", "N14", "O16"]
    tables = load_yield_tables(args.yield_root)
    ds = yt.load(latest_plotfile(args.plotdir))
    data = ds.all_data()
    mass = float(data[("StochasticStellarPop_particles", "particle_mass_at_birth")].to_value()[0])
    measured = scalar_masses(ds)

    print("test_SNII_Yields simulated/table:")
    max_error = 0.0
    for i, isotope in enumerate(isotopes):
        expected = query_fraction(tables, "SNII", isotope, mass / MSUN_CGS) * mass
        ratio = measured[i] / expected
        max_error = max(max_error, abs(ratio - 1.0))
        print(f"  {isotope:4s} scalar_{i}: simulated={measured[i]:.8e} table={expected:.8e} sim/table={ratio:.8f}")
    return max_error


def validate_wr_agb(args):
    isotopes = ["C12", "O16", "Fe56"]
    tables = load_yield_tables(args.yield_root)
    ds = yt.load(latest_plotfile(args.plotdir))
    data = ds.all_data()
    stages = data[("StochasticStellarPop_particles", "particle_evolution_stage")].to_value()
    masses = data[("StochasticStellarPop_particles", "particle_mass_at_birth")].to_value()
    birth_times = data[("StochasticStellarPop_particles", "particle_birth_time")].to_value()
    death_times = data[("StochasticStellarPop_particles", "particle_death_time")].to_value()
    wr_index = np.where(masses / MSUN_CGS >= 9.0)[0][0]
    wr_mass = float(masses[wr_index])
    agb_candidates = np.where(masses / MSUN_CGS <= 8.0)[0]
    agb_mass = float(masses[agb_candidates[0]]) if len(agb_candidates) > 0 else 7.0 * MSUN_CGS
    elapsed = float(ds.current_time)
    wr_lifetime = max(float(death_times[wr_index] - birth_times[wr_index]), 0.0)
    wr_elapsed = min(elapsed, wr_lifetime)
    if wr_lifetime <= 0.0:
        raise SystemExit("WR particle has non-positive lifetime")
    measured_total = scalar_masses(ds)
    measured_snii = scalar_masses(ds, 3)
    measured_wr = scalar_masses(ds, 6)
    measured_agb = scalar_masses(ds, 9)

    print("test_WR_AGB_yields simulated/table:")
    max_error = 0.0
    for i, isotope in enumerate(isotopes):
        wr_expected = query_fraction(tables, "WR", isotope, wr_mass / MSUN_CGS) * wr_mass * wr_elapsed / wr_lifetime
        agb_expected = query_fraction(tables, "AGB", isotope, agb_mass / MSUN_CGS) * agb_mass
        expected = wr_expected + agb_expected
        ratio = measured_total[i] / expected if expected > 0.0 else 1.0
        wr_ratio = measured_wr[i] / wr_expected if wr_expected > 0.0 else 1.0
        agb_ratio = measured_agb[i] / agb_expected if agb_expected > 0.0 else 1.0
        snii_abs = abs(measured_snii[i])
        max_error = max(max_error, abs(ratio - 1.0))
        max_error = max(max_error, abs(wr_ratio - 1.0))
        max_error = max(max_error, abs(agb_ratio - 1.0))
        max_error = max(max_error, snii_abs / max(expected, 1.0))
        print(
            f"  {isotope:4s} total scalar_{i}: simulated={measured_total[i]:.8e} table={expected:.8e} sim/table={ratio:.8f} "
            f"WR scalar_{6 + i}: simulated={measured_wr[i]:.8e} table={wr_expected:.8e} sim/table={wr_ratio:.8f} "
            f"AGB scalar_{9 + i}: simulated={measured_agb[i]:.8e} table={agb_expected:.8e} sim/table={agb_ratio:.8f} "
            f"SNII scalar_{3 + i}: simulated={measured_snii[i]:.8e}"
        )
    return max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["snii", "wr_agb"], required=True)
    parser.add_argument("--plotdir", required=True)
    parser.add_argument("--yield-root", default="extern/yields")
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    args = parser.parse_args()

    max_error = validate_snii(args) if args.case == "snii" else validate_wr_agb(args)
    if max_error > args.rtol:
        raise SystemExit(f"yield validation failed: max |ratio - 1| = {max_error:.3e} > {args.rtol:.3e}")


if __name__ == "__main__":
    main()
