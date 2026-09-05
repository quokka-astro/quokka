#!/usr/bin/env python3
"""Generate a Quokka stellar luminosity table (CSV) from slug2 isochrones.

The script runs the slug2 ``write_isochrone`` utility on a (time, initial mass)
grid with one top-hat filter per requested photometric band, writes the raw
output to a temporary text file, and converts that file into the CSV format read
by ``quokka::DataTable<2, nGroups>::CSVReader`` (see docs/markdown/data_table.md).

The resulting table is used by setting, in the Quokka input file::

    particles.rad_table = "../inputs/my_table.csv"
    particles.rad_table_output_transform = "linear"

``write_isochrone`` samples both the mass and the time axis logarithmically, which
matches the ``log`` coordinate spacing written into the CSV header.

Bands may be given either as photon-energy ranges (``--eV``) or as wavelength
ranges in Angstroms (``--lambda``); both options may be repeated, and the bands
appear in the table in the order the options are given. Quokka stores radiation
groups as a single strictly increasing list of energy edges, so the bands must
be given in order of increasing photon energy and adjacent bands must share a
boundary.

The location of the slug2 installation is taken from ``--slug-path``, or from the
``slug2_path`` or ``SLUG_DIR`` environment variables.

Usage:
    export slug2_path="/path/to/slug2"
    slug_luminosity_table_for_quokka.py PE-and-LW.csv --eV 6 11.2 --eV 11.2 13.6
    slug_luminosity_table_for_quokka.py single-band.csv --lambda 1000 1200
    slug_luminosity_table_for_quokka.py FUV-and-LyC.csv --eV 6 13.6 --eV 13.6 54.4 \
        --m0 2.1 --m1 120 --nm 21 --t0 1e5 --t1 1e8 --nt 31
    slug_luminosity_table_for_quokka.py LW.csv --eV 11.2 13.6 --slug-path /path/to/slug2
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable

import numpy as np

# h*c in eV Angstrom (CODATA 2018), used to convert photon energies to wavelengths
HC_EV_ANGSTROM = 12398.419843320026


class BandAction(argparse.Action):
    """Collect --eV and --lambda bands into a single list that preserves their order."""

    def __call__(self, parser, namespace, values, option_string=None):
        lo, hi = float(values[0]), float(values[1])
        if not hi > lo:
            parser.error(f"{option_string} needs an increasing pair, got {lo} {hi}")
        if lo <= 0.0:
            parser.error(f"{option_string} needs positive values, got {lo}")
        if namespace.bands is None:
            namespace.bands = []
        namespace.bands.append((option_string, lo, hi))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Quokka luminosity table (CSV) from slug2 isochrones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("output", help="path of the CSV table to write")
    parser.add_argument(
        "--eV",
        nargs=2,
        metavar=("E_MIN", "E_MAX"),
        action=BandAction,
        dest="bands",
        default=None,
        help="photon-energy band in eV; may be repeated",
    )
    parser.add_argument(
        "--lambda",
        nargs=2,
        metavar=("LAMBDA_MIN", "LAMBDA_MAX"),
        action=BandAction,
        dest="bands",
        help="wavelength band in Angstroms; may be repeated",
    )
    parser.add_argument("--m0", type=float, default=2.1, help="minimum initial mass, in Msun")
    parser.add_argument("--m1", type=float, default=120.0, help="maximum initial mass, in Msun")
    parser.add_argument("--nm", type=int, default=100, help="number of mass points")
    parser.add_argument("--t0", type=float, default=1.0e5, help="minimum stellar age, in yr")
    parser.add_argument("--t1", type=float, default=2.0e8, help="maximum stellar age, in yr")
    parser.add_argument("--nt", type=int, default=100, help="number of age points")
    parser.add_argument("--trackset", default="mist_2016_vvcrit_40", help="slug2 stellar track set")
    parser.add_argument("-Z", "--metallicity", type=float, default=1.0, help="metallicity relative to solar")
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="lower bound applied to every luminosity; use a small positive value for a log output transform",
    )
    parser.add_argument(
        "--slug-path",
        default=os.environ.get("slug2_path") or os.environ.get("SLUG_DIR"),
        help="slug2 installation directory (default: $slug2_path or $SLUG_DIR)",
    )
    parser.add_argument("--keep-raw", metavar="PATH", help="also save the raw write_isochrone output to PATH")

    args = parser.parse_args(argv)

    if not args.bands:
        parser.error("at least one band is required; use --eV or --lambda")
    check_quokka_band_sequence(args.bands, parser.error)
    if args.slug_path is None:
        parser.error("slug2 path is unknown; set $slug2_path or pass --slug-path")
    if args.nm < 2 or args.nt < 2:
        parser.error("--nm and --nt must both be at least 2")
    if not args.m1 > args.m0 > 0.0:
        parser.error("--m0 and --m1 must be positive with --m1 > --m0")
    if not args.t1 > args.t0 > 0.0:
        parser.error("--t0 and --t1 must be positive with --t1 > --t0")

    return args


def band_energy_eV(option: str, lo: float, hi: float) -> tuple[float, float]:
    """Return the photon-energy interval (E_min, E_max) in eV for one band."""
    if option == "--eV":
        return lo, hi
    return HC_EV_ANGSTROM / hi, HC_EV_ANGSTROM / lo


def check_quokka_band_sequence(bands: list[tuple[str, float, float]], error: Callable[[str], None]) -> None:
    """Abort unless the bands match Quokka's radiation-group energy edges.

    Quokka stores ``nGroups + 1`` strictly increasing boundaries, so later groups
    must have higher photon energy and adjacent groups must share an edge.
    """
    energies = [band_energy_eV(option, lo, hi) for option, lo, hi in bands]
    for i in range(len(energies) - 1):
        e_lo, e_hi = energies[i]
        next_lo, next_hi = energies[i + 1]
        if next_lo < e_lo or next_hi < e_hi:
            error(
                "band sequence must go with increasing energy: "
                f"group {i} is {e_lo:g}-{e_hi:g} eV but group {i + 1} is {next_lo:g}-{next_hi:g} eV"
            )
        if not math.isclose(e_hi, next_lo, rel_tol=1.0e-9, abs_tol=0.0):
            error(
                "band boundaries must be connected: "
                f"group {i} ends at {e_hi:g} eV but group {i + 1} starts at {next_lo:g} eV"
            )


def bands_to_wavelengths(bands: list[tuple[str, float, float]]) -> list[tuple[float, float]]:
    """Convert each requested band to a (lambda_min, lambda_max) pair in Angstroms."""
    wavelengths = []
    for option, lo, hi in bands:
        if option == "--eV":
            wavelengths.append((HC_EV_ANGSTROM / hi, HC_EV_ANGSTROM / lo))
        else:
            wavelengths.append((lo, hi))
    return wavelengths


def run_write_isochrone(args: argparse.Namespace, wavelengths: list[tuple[float, float]], raw_path: str) -> None:
    """Run slug2's write_isochrone and store its stdout in raw_path."""
    executable = os.path.join(args.slug_path, "bin", "write_isochrone")
    if not os.path.isfile(executable):
        sys.exit(f"error: {executable} not found; check --slug-path")

    command = [executable, "-Z", repr(args.metallicity)]
    command += ["-m0", repr(args.m0), "-m1", repr(args.m1), "-nm", str(args.nm)]
    command += ["-t0", repr(args.t0), "-t1", repr(args.t1), "-nt", str(args.nt)]
    for lambda_min, lambda_max in wavelengths:
        command += ["-tf", repr(lambda_min), repr(lambda_max)]
    command.append(args.trackset)

    # write_isochrone locates lib/tracks and lib/atmospheres relative to $SLUG_DIR
    environment = dict(os.environ, SLUG_DIR=args.slug_path)
    print("running: " + " ".join(command), file=sys.stderr)
    with open(raw_path, "w", encoding="utf-8") as raw_file:
        result = subprocess.run(command, stdout=raw_file, env=environment, check=False)
    if result.returncode != 0:
        sys.exit(f"error: write_isochrone exited with status {result.returncode}")


def parse_isochrones(raw_path: str, n_bands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse write_isochrone output into (times, masses, luminosities).

    Returns times with shape (nt,), masses with shape (nm,), and luminosities with
    shape (nt, nm, n_bands) in erg/s. Entries marked "--" are stars that no longer
    exist at that age, and are returned as zero.
    """
    times: list[float] = []
    masses_per_block: list[list[float]] = []
    lum_per_block: list[list[list[float]]] = []

    with open(raw_path, encoding="utf-8") as raw_file:
        for raw_line in raw_file:
            line = raw_line.strip()
            if not line or line.startswith("---") or line.startswith("["):
                continue  # separator or units line
            tokens = line.split()
            if tokens[0] == "time":
                times.append(float(tokens[1]))
                masses_per_block.append([])
                lum_per_block.append([])
                continue
            if tokens[0] == "m_init":
                continue  # column-name line
            if not times:
                sys.exit(f"error: unexpected line before the first isochrone: {line}")
            if len(tokens) < n_bands + 1:
                sys.exit(f"error: expected at least {n_bands + 1} columns, got: {line}")
            masses_per_block[-1].append(float(tokens[0]))
            # A dead star is printed as "--" in every column; a band with negligible
            # luminosity can come out slightly negative from the numerical integration.
            lum_per_block[-1].append([0.0 if token == "--" else float(token) for token in tokens[-n_bands:]])

    if not times:
        sys.exit("error: write_isochrone produced no isochrones")
    n_mass = len(masses_per_block[0])
    for time, block in zip(times, masses_per_block):
        if block != masses_per_block[0]:
            sys.exit(f"error: the mass grid at time {time} differs from the first isochrone")
    if n_mass < 2:
        sys.exit("error: fewer than two mass points were returned")

    return np.array(times), np.array(masses_per_block[0]), np.array(lum_per_block)


def check_grid(name: str, values: np.ndarray, lo: float, hi: float, count: int) -> None:
    """Verify that slug2 sampled the axis logarithmically between lo and hi."""
    if values.size != count:
        sys.exit(f"error: expected {count} {name} points, got {values.size}")
    expected = np.geomspace(lo, hi, count)
    # write_isochrone prints only 6 significant figures, hence the loose tolerance
    if not np.allclose(values, expected, rtol=1.0e-5):
        sys.exit(f"error: the {name} grid is not log-spaced between {lo} and {hi}")


def write_csv(path: str, args: argparse.Namespace, luminosities: np.ndarray) -> None:
    """Write the table in the CSV format read by quokka::DataTable<2, Nout>::CSVReader."""
    n_time, n_mass, n_bands = luminosities.shape
    lines = [
        "2",
        f"{n_time},{n_mass}",
        f"{n_bands}",
        "age,mass",
        ",".join(f"luminosity_group{group}" for group in range(n_bands)),
        "year,Msun",
        ",".join(["erg/s"] * n_bands),
        f"{args.t0!r},{args.m0!r}",
        f"{args.t1!r},{args.m1!r}",
        "log,log",
    ]
    # For each output, the rows run over the second input (mass) and the columns
    # over the first input (age).
    for group in range(n_bands):
        for i_mass in range(n_mass):
            lines.append(",".join(f"{value:.8e}" for value in luminosities[:, i_mass, group]))

    with open(path, "w", encoding="utf-8") as csv_file:
        csv_file.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    wavelengths = bands_to_wavelengths(args.bands)

    raw_path = args.keep_raw
    with tempfile.TemporaryDirectory() as scratch_dir:
        if raw_path is None:
            raw_path = os.path.join(scratch_dir, "isochrones.txt")
        run_write_isochrone(args, wavelengths, raw_path)
        times, masses, luminosities = parse_isochrones(raw_path, len(wavelengths))

    check_grid("age", times, args.t0, args.t1, args.nt)
    check_grid("mass", masses, args.m0, args.m1, args.nm)

    n_negative = int(np.count_nonzero(luminosities < args.floor))
    luminosities = np.maximum(luminosities, args.floor)
    write_csv(args.output, args, luminosities)

    print(f"wrote {args.output}: {args.nt} ages x {args.nm} masses x {len(wavelengths)} bands")
    for group, ((option, lo, hi), (lambda_min, lambda_max)) in enumerate(zip(args.bands, wavelengths)):
        unit = "eV" if option == "--eV" else "Angstrom"
        print(f"  luminosity_group{group}: {lo:g} - {hi:g} {unit} ({lambda_min:.4f} - {lambda_max:.4f} Angstrom)")
    print(f"  raised {n_negative} entries to the floor of {args.floor:g} erg/s")


if __name__ == "__main__":
    main()
