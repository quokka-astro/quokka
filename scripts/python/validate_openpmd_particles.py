#!/usr/bin/env python3
"""Validate openPMD particle output against Quokka CSV snapshots."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import openpmd_api as io

try:
    RecordComponent = io.RecordComponent
except AttributeError:  # Compatibility with older openPMD-api releases
    RecordComponent = io.Record_Component

AXES: Tuple[str, ...] = ("x", "y", "z")
SPECIES_FAMILIES: Dict[str, str] = {
    "Rad_particles": "rad",
    "CIC_particles": "cic",
    "CICRad_particles": "cicrad",
    "StochasticStellarPop_particles": "stochastic",
    "Sink_particles": "sink",
    "Test_particles": "test",
}


@dataclass
class CsvParticleData:
    ids: np.ndarray
    positions: np.ndarray
    real: np.ndarray
    ints: np.ndarray
    real_headers: List[str]
    int_headers: List[str]


@dataclass
class OpenPMDParticleData:
    ids: np.ndarray
    positions: np.ndarray
    real: np.ndarray
    ints: np.ndarray


class ParticleComparisonError(RuntimeError):
    """Raised when validation fails for a particle species."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare particle data stored in openPMD files with Quokka CSV exports.",
    )
    parser.add_argument(
        "--openpmd",
        required=True,
        type=Path,
        help="Path to the openPMD file (e.g. plt00042.bp).",
    )
    parser.add_argument(
        "--csv-root",
        default=Path("."),
        type=Path,
        help="Directory containing partXXXXX folders (defaults to current directory).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration index to read from the openPMD series (auto-detected from filename if omitted).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for floating-point comparisons (default: 1e-12).",
    )
    parser.add_argument(
        "--skip-missing-csv",
        action="store_true",
        help="Skip species that exist in openPMD output but lack a corresponding CSV file.",
    )
    return parser.parse_args()


def detect_iteration_from_path(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.name)
    if not matches:
        raise ParticleComparisonError(
            f"Unable to infer iteration number from openPMD filename '{path.name}'. Provide --iteration explicitly."
        )
    return int(matches[-1])


def load_csv_particles(csv_path: Path) -> CsvParticleData:
    if not csv_path.is_file():
        raise ParticleComparisonError(f"Missing CSV file: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ParticleComparisonError(f"CSV file '{csv_path}' has no header row.")

        headers = reader.fieldnames
        pos_headers = sorted((h for h in headers if h.startswith("pos_")), key=lambda h: int(h.split("_")[1]))
        if not pos_headers:
            raise ParticleComparisonError(
                f"CSV file '{csv_path}' is missing position columns (expected pos_0, pos_1, ...)."
            )
        real_headers = sorted((h for h in headers if h.startswith("real_")), key=lambda h: int(h.split("_")[1]))
        int_headers = sorted((h for h in headers if h.startswith("int_")), key=lambda h: int(h.split("_")[1]))

        ids: List[int] = []
        positions: List[List[float]] = []
        real_data: List[List[float]] = []
        int_data: List[List[int]] = []

        for row in reader:
            ids.append(int(row["ID"]))
            positions.append([float(row[h]) for h in pos_headers])
            if real_headers:
                real_data.append([float(row[h]) for h in real_headers])
            if int_headers:
                int_data.append([int(row[h]) for h in int_headers])

    ids_arr = np.asarray(ids, dtype=np.int64)
    pos_arr = np.asarray(positions, dtype=np.float64) if positions else np.empty((0, len(pos_headers)), dtype=np.float64)
    real_arr = np.asarray(real_data, dtype=np.float64) if real_headers else np.empty((len(ids), 0), dtype=np.float64)
    int_arr = np.asarray(int_data, dtype=np.int64) if int_headers else np.empty((len(ids), 0), dtype=np.int64)

    return CsvParticleData(
        ids=ids_arr,
        positions=pos_arr,
        real=real_arr,
        ints=int_arr,
        real_headers=real_headers,
        int_headers=int_headers,
    )


def expected_real_components(species_name: str, dimension: int, count: int) -> List[Tuple[str, str]]:
    family = SPECIES_FAMILIES.get(species_name)
    if family is None:
        raise ParticleComparisonError(f"Unsupported particle species '{species_name}'. Update SPECIES_FAMILIES map.")

    components: List[Tuple[str, str]] = []
    if family == "rad":
        components.extend([("birthTime", "scalar"), ("deathTime", "scalar")])
        luminosity = count - len(components)
        if luminosity < 0:
            raise ParticleComparisonError(
                f"CSV real component count ({count}) is inconsistent with expected fields for '{species_name}'."
            )
        components.extend(("luminosity", f"g{idx}") for idx in range(luminosity))
    elif family in {"cic", "sink"}:
        components.append(("mass", "scalar"))
        components.extend(("velocity", axis) for axis in AXES[:dimension])
    elif family in {"cicrad", "stochastic", "test"}:
        components.append(("mass", "scalar"))
        components.extend(("velocity", axis) for axis in AXES[:dimension])
        components.extend([("birthTime", "scalar"), ("deathTime", "scalar")])
        luminosity = count - len(components)
        if luminosity < 0:
            raise ParticleComparisonError(
                f"CSV real component count ({count}) is inconsistent with expected fields for '{species_name}'."
            )
        components.extend(("luminosity", f"g{idx}") for idx in range(luminosity))
    else:
        raise ParticleComparisonError(f"Unhandled species family '{family}'.")

    if len(components) != count:
        raise ParticleComparisonError(
            f"CSV real component count ({count}) does not match expected components ({len(components)}) for '{species_name}'."
        )
    return components


def expected_int_components(species_name: str, count: int) -> List[Tuple[str, str]]:
    family = SPECIES_FAMILIES.get(species_name)
    if family is None:
        raise ParticleComparisonError(f"Unsupported particle species '{species_name}'.")

    components: List[Tuple[str, str]] = []
    if family in {"stochastic", "test"}:
        components.append(("evolutionStage", "scalar"))

    if len(components) != count:
        raise ParticleComparisonError(
            f"CSV integer component count ({count}) does not match expected components ({len(components)}) for '{species_name}'."
        )
    return components


def schedule_component(
    species: io.ParticleSpecies,
    species_label: str,
    record_name: str,
    label: str,
    schedule,
) -> np.ndarray:
    try:
        record = species[record_name]
    except KeyError as exc:
        raise ParticleComparisonError(
            f"Record '{record_name}' not found for particle species '{species_label}'."
        ) from exc

    component_key: object
    if label == "scalar":
        component_key = RecordComponent.SCALAR
    else:
        component_key = label

    try:
        component = record[component_key]
    except KeyError as exc:
        raise ParticleComparisonError(
            f"Component '{label}' missing in record '{record_name}' for particle species '{species_label}'."
        ) from exc

    return schedule(component)


def load_openpmd_particles(
    series: io.Series,
    iteration: io.Iteration,
    species_name: str,
    csv_data: CsvParticleData,
) -> OpenPMDParticleData:
    species = iteration.particles[species_name]
    requests: List[np.ndarray] = []

    def schedule(component) -> np.ndarray:
        data = component.load_chunk()
        requests.append(data)
        return data

    # IDs
    id_ref = schedule(species["id"][RecordComponent.SCALAR])

    # Positions (dimension inferred from CSV columns)
    dimension = csv_data.positions.shape[1]
    position_refs: List[np.ndarray] = []
    for axis in AXES[:dimension]:
        try:
            comp = species["position"][axis]
        except KeyError as exc:
            raise ParticleComparisonError(
                f"Species '{species_name}' is missing position component '{axis}'."
            ) from exc
        position_refs.append(schedule(comp))

    # Reals
    real_expected = expected_real_components(species_name, dimension, csv_data.real.shape[1])
    real_refs = [schedule_component(species, species_name, record, label, schedule) for record, label in real_expected]

    # Ints
    int_expected = expected_int_components(species_name, csv_data.ints.shape[1])
    int_refs = [schedule_component(species, species_name, record, label, schedule) for record, label in int_expected]

    # Execute pending IO
    series.flush()

    def to_array(ref: np.ndarray, dtype: np.dtype) -> np.ndarray:
        arr = np.array(ref, copy=False)
        if arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr

    ids_raw = to_array(id_ref, np.uint64)
    if ids_raw.ndim != 1:
        raise ParticleComparisonError(f"Particle IDs for '{species_name}' are not one-dimensional.")

    int64_max = np.uint64(np.iinfo(np.int64).max)
    if np.any(ids_raw > int64_max):
        raise ParticleComparisonError(
            f"Particle IDs for '{species_name}' exceed signed 64-bit range (max {int(int64_max)})."
        )
    ids = ids_raw.astype(np.int64, copy=False)

    if position_refs:
        position_arrays = [to_array(ref, np.float64) for ref in position_refs]
        positions = np.stack(position_arrays, axis=1)
    else:
        positions = np.empty((ids.size, 0), dtype=np.float64)

    if real_refs:
        real_arrays = [to_array(ref, np.float64) for ref in real_refs]
        real = np.stack(real_arrays, axis=1)
    else:
        real = np.empty((ids.size, 0), dtype=np.float64)

    if int_refs:
        int_arrays = [to_array(ref, np.int64) for ref in int_refs]
        ints = np.stack(int_arrays, axis=1)
    else:
        ints = np.empty((ids.size, 0), dtype=np.int64)

    return OpenPMDParticleData(ids=ids, positions=positions, real=real, ints=ints)


@dataclass
class SpeciesComparisonResult:
    species: str
    particle_count: int
    max_position_error: Sequence[float]
    max_real_error: Sequence[float]
    int_mismatches: int
    passed: bool


def compare_species(
    species_name: str,
    csv_data: CsvParticleData,
    openpmd_data: OpenPMDParticleData,
    tol: float,
) -> SpeciesComparisonResult:
    csv_ids_int = csv_data.ids.astype(np.int64, copy=False)
    open_ids_int = openpmd_data.ids.astype(np.int64, copy=False)

    open_nonpositive = int(np.count_nonzero(open_ids_int <= 0))
    open_unique = np.unique(open_ids_int)
    open_duplicates = int(open_ids_int.size - open_unique.size)

    if csv_ids_int.size != open_ids_int.size:
        raise ParticleComparisonError(
            f"Particle count mismatch for '{species_name}': CSV has {csv_ids_int.size}, openPMD has {open_ids_int.size}."
        )

    if csv_ids_int.size == 0:
        return SpeciesComparisonResult(species_name, 0, [], [], 0, True)

    csv_order = np.argsort(csv_ids_int)
    open_order = np.argsort(open_ids_int)

    sorted_csv_ids = csv_ids_int[csv_order]
    sorted_open_ids = open_ids_int[open_order]
    if not np.array_equal(sorted_csv_ids, sorted_open_ids):
        csv_set = set(sorted_csv_ids.tolist())
        open_set = set(sorted_open_ids.tolist())

        sample_csv = sorted(list(csv_set - open_set))[:5]
        sample_open = sorted(list(open_set - csv_set))[:5]
        details = [
            f"Particle ID mismatch for '{species_name}'. Sample IDs only in CSV: {sample_csv}; only in openPMD: {sample_open}.",
        ]
        if open_nonpositive:
            nonpositive_examples = np.sort(open_ids_int[open_ids_int <= 0])[:5].tolist()
            details.append(
                f"openPMD reported {open_nonpositive} non-positive IDs (examples: {nonpositive_examples})."
            )
        if open_duplicates:
            details.append(f"openPMD reported {open_duplicates} duplicate IDs.")
        raise ParticleComparisonError(" ".join(details))

    csv_positions = csv_data.positions[csv_order]
    open_positions = openpmd_data.positions[open_order]
    if csv_positions.shape != open_positions.shape:
        raise ParticleComparisonError(
            f"Position array shape mismatch for '{species_name}': CSV {csv_positions.shape} vs openPMD {open_positions.shape}."
        )

    if csv_positions.size:
        pos_diff = np.abs(open_positions - csv_positions)
        max_pos = pos_diff.max(axis=0)
    else:
        max_pos = np.array([], dtype=float)

    csv_real = csv_data.real[csv_order]
    open_real = openpmd_data.real[open_order]
    if csv_real.shape != open_real.shape:
        raise ParticleComparisonError(
            f"Real component shape mismatch for '{species_name}': CSV {csv_real.shape} vs openPMD {open_real.shape}."
        )

    if csv_real.size:
        real_diff = np.abs(open_real - csv_real)
        max_real = real_diff.max(axis=0)
    else:
        max_real = np.array([], dtype=float)

    csv_ints = csv_data.ints[csv_order]
    open_ints = openpmd_data.ints[open_order]
    if csv_ints.shape != open_ints.shape:
        raise ParticleComparisonError(
            f"Integer component shape mismatch for '{species_name}': CSV {csv_ints.shape} vs openPMD {open_ints.shape}."
        )

    if csv_ints.size:
        int_mismatches = int(np.sum(open_ints != csv_ints))
    else:
        int_mismatches = 0

    passed = (
        (max_pos.size == 0 or np.all(max_pos <= tol))
        and (max_real.size == 0 or np.all(max_real <= tol))
        and int_mismatches == 0
    )

    return SpeciesComparisonResult(
        species=species_name,
        particle_count=int(csv_ids_int.size),
        max_position_error=max_pos.tolist(),
        max_real_error=max_real.tolist(),
        int_mismatches=int_mismatches,
        passed=passed,
    )


def main() -> int:
    args = parse_args()
    openpmd_path: Path = args.openpmd
    csv_root: Path = args.csv_root

    if not openpmd_path.exists():
        raise ParticleComparisonError(f"openPMD file not found: {openpmd_path}")

    iteration_index = args.iteration if args.iteration is not None else detect_iteration_from_path(openpmd_path)
    part_dir = csv_root / f"part{iteration_index:05d}"
    if not part_dir.is_dir():
        raise ParticleComparisonError(f"CSV directory '{part_dir}' does not exist.")

    csv_files = {path.stem: path for path in part_dir.glob("*.csv")}
    if not csv_files:
        raise ParticleComparisonError(f"No CSV particle files found in '{part_dir}'.")

    series = io.Series(str(openpmd_path), io.Access.read_only)

    if iteration_index not in series.iterations:
        if len(series.iterations) == 1 and args.iteration is None:
            iteration_index = next(iter(series.iterations.keys()))
        else:
            available = sorted(series.iterations.keys())
            raise ParticleComparisonError(
                f"Iteration {iteration_index} not found in openPMD file. Available iterations: {available}."
            )

    iteration = series.iterations[iteration_index]
    try:
        iteration.open()
    except AttributeError:
        pass

    processed_species: List[SpeciesComparisonResult] = []
    failures: List[str] = []

    for species_name in iteration.particles:
        if species_name not in csv_files:
            message = (
                f"Species '{species_name}' present in openPMD iteration {iteration_index} but no matching CSV file found."
            )
            if args.skip_missing_csv:
                print(f"[WARN] {message}")
                continue
            failures.append(message)
            continue

        csv_data = load_csv_particles(csv_files[species_name])
        open_data = load_openpmd_particles(series, iteration, species_name, csv_data)
        result = compare_species(species_name, csv_data, open_data, args.tolerance)
        processed_species.append(result)
        status = "PASS" if result.passed else "FAIL"
        pos_err = ", ".join(f"{err:.3e}" for err in result.max_position_error) or "-"
        real_err = ", ".join(f"{err:.3e}" for err in result.max_real_error) or "-"
        print(
            f"[{status}] {species_name}: N={result.particle_count}, "
            f"max|Δpos|={pos_err}, max|Δreal|={real_err}, Δint={result.int_mismatches}"
        )
        if not result.passed:
            failures.append(
                f"Species '{species_name}' exceeded tolerance (positions {result.max_position_error}, "
                f"real {result.max_real_error}, int mismatches {result.int_mismatches})."
            )

    extra_csv = sorted(set(csv_files.keys()) - {res.species for res in processed_species})
    for extra in extra_csv:
        print(f"[INFO] CSV file '{csv_files[extra].name}' has no corresponding openPMD species. Skipping.")

    try:
        iteration.close()
    except AttributeError:
        pass
    series.close()

    if failures:
        for msg in failures:
            print(f"[ERROR] {msg}", file=sys.stderr)
        return 1

    if not processed_species:
        print("No particle species compared. Nothing to validate.")
        return 0

    print("All compared particle species passed within the specified tolerance.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ParticleComparisonError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
