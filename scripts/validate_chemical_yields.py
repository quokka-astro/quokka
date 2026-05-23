#!/usr/bin/env python3
"""Validate gas scalar masses against tabulated chemical yields."""

import argparse
import glob
import math
import re
from pathlib import Path

import numpy as np
import yt

MSUN_CGS = 1.9884e33
WR_AGB_WINDOW = 1.0e20


def normalize_numeric(token):
    return token.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", "")


def parse_metallicity_folder(name):
    pos = name.find("models")
    if len(name) < 7 or not name.startswith("z") or pos <= 1:
        return -1.0
    digits = name[1:pos]
    return int(digits) / (10 ** len(digits))


def add_entry(entries, mass, metallicity):
    key = (round(mass * 1000.0), round(metallicity * 1.0e6))
    if key not in entries:
        entries[key] = [0.0] * 9
    return entries[key]


def load_yields(root, isotopes):
    root = Path(root)
    iso_index = {name.lower(): i for i, name in enumerate(isotopes)}
    entries = {}
    mass_re = re.compile(r"s([0-9]+(?:\.[0-9]+)?)\.yield_table")

    sukhbold_root = root / "SNII_Sukhbold16"
    if sukhbold_root.is_dir():
        for table in sukhbold_root.iterdir():
            match = mass_re.fullmatch(table.name)
            if not (table.is_file() and match):
                continue
            mass = float(match.group(1))
            values = add_entry(entries, mass, 0.014)
            has_ejecta_col = False
            has_wind_col = False
            for line in table.read_text(errors="ignore").splitlines():
                if not line or line[0] == "#":
                    continue
                if line[0] == "[":
                    has_ejecta_col = "[ejecta]" in line
                    has_wind_col = "[wind]" in line
                    continue
                parts = line.split()
                if not parts or parts[0].lower() not in iso_index:
                    continue
                isotope_index = iso_index[parts[0].lower()]
                col = 1
                if has_ejecta_col:
                    if len(parts) <= col:
                        continue
                    values[isotope_index] += max(float(normalize_numeric(parts[col])) / mass, 0.0)
                    col += 1
                if has_wind_col:
                    if len(parts) <= col:
                        continue
                    values[3 + isotope_index] += max(float(normalize_numeric(parts[col])) / mass, 0.0)

    agb_root = root / "AGB_Karakas16"
    filename_re = re.compile(r"m([0-9]+(?:\.[0-9]+)?)z([0-9]+).*\.dat", re.IGNORECASE)
    header_re = re.compile(
        r"#\s*Initial\s+mass\s*=\s*([0-9]+(?:\.[0-9]+)?),\s*Z\s*=\s*([0-9]+(?:\.[0-9]+)?),.*M_mix\s*=\s*([0-9eE+\-.]+)",
        re.IGNORECASE,
    )
    if agb_root.is_dir():
        for table in agb_root.rglob("*.dat"):
            text = None
            mass = -1.0
            metallicity = -1.0
            match = filename_re.fullmatch(table.name)
            if match:
                mass = float(match.group(1))
                z_digits = match.group(2)
                metallicity = int(z_digits) / (10 ** len(z_digits))
            else:
                text = table.read_text(errors="ignore")
                for line in text.splitlines():
                    header = header_re.search(line)
                    if header:
                        mass = float(header.group(1))
                        metallicity = float(header.group(2))
                        break
            if mass <= 0.0 or metallicity <= 0.0:
                continue
            if text is None:
                text = table.read_text(errors="ignore")
            values = add_entry(entries, mass, metallicity)
            for line in text.splitlines():
                if not line or line[0] == "#":
                    continue
                parts = line.split()
                if len(parts) < 3 or parts[0].lower() not in iso_index:
                    continue
                values[6 + iso_index[parts[0].lower()]] += max(float(normalize_numeric(parts[2])) / mass, 0.0)

    doherty_root = root / "superAGB_Doherty14"
    header_re = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)M\s+Z=([0-9eE+\-.]+).*", re.IGNORECASE)
    if doherty_root.is_dir():
        for table in doherty_root.iterdir():
            if not table.is_file():
                continue
            mass = -1.0
            values = None
            for line in table.read_text(errors="ignore").splitlines():
                header = header_re.fullmatch(line)
                if header:
                    mass = float(header.group(1))
                    values = add_entry(entries, mass, float(header.group(2)))
                    continue
                if values is None or mass <= 0.0 or not line or line[0] == "#":
                    continue
                parts = line.split()
                if len(parts) < 2 or parts[0].lower() not in iso_index:
                    continue
                values[6 + iso_index[parts[0].lower()]] += max(float(normalize_numeric(parts[1])) / mass, 0.0)

    return [(max(k[0] / 1000.0, 1.0e-12), max(k[1] / 1.0e6, 1.0e-12), values) for k, values in entries.items()]


def query_fraction(entries, channel, isotope_index, mass_msun, metallicity):
    log_mass = math.log10(max(mass_msun, 1.0e-12))
    log_z = math.log10(max(metallicity, 1.0e-12))
    weighted_sum = 0.0
    weight_total = 0.0
    flat_index = 3 * channel + isotope_index

    for entry_mass, entry_z, values in entries:
        frac = values[flat_index]
        if frac <= 0.0:
            continue
        dist2 = (math.log10(entry_mass) - log_mass) ** 2 + (math.log10(entry_z) - log_z) ** 2
        if dist2 < 1.0e-10:
            return frac
        weight = 1.0 / (dist2 + 1.0e-20)
        weighted_sum += weight * frac
        weight_total += weight

    return weighted_sum / weight_total if weight_total > 0.0 else 0.0


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
    entries = load_yields(args.yield_root, isotopes)
    ds = yt.load(latest_plotfile(args.plotdir))
    data = ds.all_data()
    mass = float(data[("StochasticStellarPop_particles", "particle_mass_at_birth")].to_value()[0])
    measured = scalar_masses(ds)

    print("test_SNII_Yields measured/expected:")
    max_error = 0.0
    for i, isotope in enumerate(isotopes):
        expected = query_fraction(entries, 0, i, mass / MSUN_CGS, 0.014) * mass
        ratio = measured[i] / expected
        max_error = max(max_error, abs(ratio - 1.0))
        print(f"  {isotope:4s} scalar_{i}: measured={measured[i]:.8e} expected={expected:.8e} ratio={ratio:.8f}")
    return max_error


def validate_wr_agb(args):
    isotopes = ["C12", "O16", "Fe56"]
    entries = load_yields(args.yield_root, isotopes)
    ds = yt.load(latest_plotfile(args.plotdir))
    data = ds.all_data()
    stages = data[("StochasticStellarPop_particles", "particle_evolution_stage")].to_value()
    masses = data[("StochasticStellarPop_particles", "particle_mass_at_birth")].to_value()
    birth_times = data[("StochasticStellarPop_particles", "particle_birth_time")].to_value()
    death_times = data[("StochasticStellarPop_particles", "particle_death_time")].to_value()
    wr_index = np.where(stages == 0)[0][0]
    wr_mass = float(masses[wr_index])
    agb_mass = float(masses[np.where(stages == 3)[0][0]])
    elapsed = float(ds.current_time)
    wr_lifetime = max(float(death_times[wr_index] - birth_times[wr_index]), 0.0)
    wr_elapsed = min(elapsed, wr_lifetime)
    if wr_lifetime <= 0.0:
        raise SystemExit("WR particle has non-positive lifetime")
    measured_total = scalar_masses(ds)
    measured_snii = scalar_masses(ds, 3)
    measured_wr = scalar_masses(ds, 6)
    measured_agb = scalar_masses(ds, 9)

    print("test_WR_AGB_yields measured/expected:")
    max_error = 0.0
    for i, isotope in enumerate(isotopes):
        wr_expected = query_fraction(entries, 1, i, wr_mass / MSUN_CGS, 0.02) * wr_mass * wr_elapsed / wr_lifetime
        agb_expected = query_fraction(entries, 2, i, agb_mass / MSUN_CGS, 0.02) * agb_mass * elapsed / WR_AGB_WINDOW
        expected = wr_expected + agb_expected
        ratio = measured_total[i] / expected
        wr_ratio = measured_wr[i] / wr_expected if wr_expected > 0.0 else 1.0
        agb_ratio = measured_agb[i] / agb_expected if agb_expected > 0.0 else 1.0
        snii_abs = abs(measured_snii[i])
        max_error = max(max_error, abs(ratio - 1.0))
        max_error = max(max_error, abs(wr_ratio - 1.0))
        max_error = max(max_error, abs(agb_ratio - 1.0))
        max_error = max(max_error, snii_abs / max(expected, 1.0))
        print(
            f"  {isotope:4s} total scalar_{i}: measured={measured_total[i]:.8e} expected={expected:.8e} ratio={ratio:.8f} "
            f"WR scalar_{6 + i}: measured={measured_wr[i]:.8e} expected={wr_expected:.8e} ratio={wr_ratio:.8f} "
            f"AGB scalar_{9 + i}: measured={measured_agb[i]:.8e} expected={agb_expected:.8e} ratio={agb_ratio:.8f} "
            f"SNII scalar_{3 + i}: measured={measured_snii[i]:.8e}"
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
