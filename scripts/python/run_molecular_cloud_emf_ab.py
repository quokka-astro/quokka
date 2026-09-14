#!/usr/bin/env python3
"""Run and compare matched molecular-cloud simulations with EMF disabled/enabled."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


MYR_SECONDS = 3.15576e13
PROBLEM = "MolecularCloudDisruption"
ARM_FLAGS = {"EMF_off": 0, "EMF_on": 1}
DISRUPTION_METRICS = (
    "cloud_dense_fraction",
    "cloud_cold_dense_fraction",
    "cloud_mass_Msun",
)
DIAGNOSTIC_METRICS = (
    "emf_active_particle_count",
    "emf_momentum_requested_step_Msun_kmps",
    "feedback_coupling_radius_pc",
    "sn_count_cumulative",
    "scalar_closure_relative_L1",
)
RESERVED_OVERRIDES = {
    "amr.max_grid_size",
    "amr.n_cell",
    "checkpoint_prefix",
    "cooling.hdf5_data_file",
    "particles.disable_SN_feedback",
    "particles.EMF_enabled",
    "plotfile_prefix",
    "problem.stellar_particles_file",
    "statistics_file",
}


@dataclass(frozen=True)
class History:
    path: Path
    rows: list[dict[str, float]]

    @property
    def final_time(self) -> float:
        return self.rows[-1]["time"]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_resolution(value: str) -> int:
    resolution = int(value)
    if resolution <= 0 or resolution % 32 != 0:
        raise argparse.ArgumentTypeError("resolution must be a positive multiple of the benchmark blocking factor (32)")
    return resolution


def validate_positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_override(value: str) -> str:
    if "=" not in value:
        raise argparse.ArgumentTypeError("overrides must have the form KEY=VALUE")
    key, override_value = value.split("=", maxsplit=1)
    key = key.strip()
    if not key or not override_value.strip():
        raise argparse.ArgumentTypeError("overrides must have a non-empty key and value")
    if key in RESERVED_OVERRIDES:
        raise argparse.ArgumentTypeError(f"'{key}' is controlled by this script and cannot be overridden")
    return f"{key}={override_value}"


def require_emf_implementation(root: Path) -> None:
    source_suffixes = {".cpp", ".h", ".hpp"}
    matches = []
    for path in (root / "src").rglob("*"):
        if path.suffix in source_suffixes and "EMF_enabled" in path.read_text(errors="replace"):
            matches.append(path)
    if not matches:
        raise SystemExit(
            "particles.EMF_enabled is not implemented in this checkout's C++ sources.\n"
            "Refusing to run two silently identical arms; rerun this script after the general EMF implementation lands."
        )


def run_checked(command: list[str], *, cwd: Path) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def git_metadata(root: Path) -> dict[str, object]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=False, capture_output=True, text=True)
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def parse_history(path: Path) -> History:
    if not path.is_file():
        raise RuntimeError(f"history file was not produced: {path}")

    columns: list[str] | None = None
    rows: list[dict[str, float]] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("# cycle time "):
            columns = line[1:].split()
            continue
        if not line or line.startswith("#") or columns is None:
            continue

        values = line.split()
        if len(values) != len(columns):
            raise RuntimeError(f"malformed row in {path}: expected {len(columns)} values, found {len(values)}")
        try:
            row = {name: float(value) for name, value in zip(columns, values)}
        except ValueError as exc:
            raise RuntimeError(f"non-numeric row in {path}: {line}") from exc
        rows.append(row)

    if not rows:
        raise RuntimeError(f"no history rows found in {path}")
    if any("time" not in row for row in rows):
        raise RuntimeError(f"history file does not contain a time column: {path}")
    if any(next_row["time"] < row["time"] for row, next_row in zip(rows, rows[1:])):
        raise RuntimeError(f"history times are not monotonically increasing: {path}")

    required = {*DISRUPTION_METRICS, *DIAGNOSTIC_METRICS}
    missing = sorted(required - rows[-1].keys())
    if missing:
        raise RuntimeError(f"history file is missing required columns {missing}: {path}")
    return History(path=path, rows=rows)


def interpolate(history: History, metric: str, target_time: float) -> float:
    if target_time <= history.rows[0]["time"]:
        return history.rows[0][metric]
    for left, right in zip(history.rows, history.rows[1:]):
        if target_time <= right["time"]:
            delta_t = right["time"] - left["time"]
            if delta_t == 0.0:
                return right[metric]
            weight = (target_time - left["time"]) / delta_t
            return left[metric] + weight * (right[metric] - left[metric])
    return history.rows[-1][metric]


def first_downward_crossing(history: History, metric: str, threshold: float) -> float | None:
    first = history.rows[0]
    if first[metric] <= threshold:
        return first["time"]
    for left, right in zip(history.rows, history.rows[1:]):
        left_value = left[metric]
        right_value = right[metric]
        if left_value > threshold >= right_value:
            delta = right_value - left_value
            if delta == 0.0:
                return right["time"]
            weight = (threshold - left_value) / delta
            return left["time"] + weight * (right["time"] - left["time"])
    return None


def relative_difference(on_value: float, off_value: float) -> float | None:
    if off_value == 0.0:
        return None
    return (on_value - off_value) / off_value


def make_comparison(histories: dict[str, History], elapsed_wall_s: dict[str, float]) -> dict[str, object]:
    off = histories["EMF_off"]
    on = histories["EMF_on"]
    common_time = min(off.final_time, on.final_time)
    initial_reference = {metric: off.rows[0][metric] for metric in DISRUPTION_METRICS}
    initial_on = {metric: on.rows[0][metric] for metric in DISRUPTION_METRICS}
    unmatched = [metric for metric in DISRUPTION_METRICS if not math.isclose(initial_reference[metric], initial_on[metric], rel_tol=1.0e-12)]
    if unmatched:
        raise RuntimeError(f"A/B arms do not have matched initial diagnostics: {unmatched}")

    arms: dict[str, object] = {}
    for name, history in histories.items():
        final_values = {metric: interpolate(history, metric, common_time) for metric in (*DISRUPTION_METRICS, *DIAGNOSTIC_METRICS)}
        t50_s = {
            metric: first_downward_crossing(history, metric, 0.5 * initial_reference[metric]) for metric in DISRUPTION_METRICS
        }
        arms[name] = {
            "EMF_enabled": ARM_FLAGS[name],
            "elapsed_wall_s": elapsed_wall_s[name],
            "history_file": str(history.path),
            "history_final_time_s": history.final_time,
            "values_at_common_end": final_values,
            "t50_s": t50_s,
            "t50_Myr": {metric: None if value is None else value / MYR_SECONDS for metric, value in t50_s.items()},
        }

    differences: dict[str, object] = {}
    off_values = arms["EMF_off"]["values_at_common_end"]  # type: ignore[index]
    on_values = arms["EMF_on"]["values_at_common_end"]  # type: ignore[index]
    for metric in (*DISRUPTION_METRICS, *DIAGNOSTIC_METRICS):
        off_value = off_values[metric]  # type: ignore[index]
        on_value = on_values[metric]  # type: ignore[index]
        differences[metric] = {
            "on_minus_off": on_value - off_value,
            "relative_on_minus_off": relative_difference(on_value, off_value),
        }

    return {
        "common_end_time_s": common_time,
        "common_end_time_Myr": common_time / MYR_SECONDS,
        "t50_reference": "0.5 times the EMF_off initial value",
        "initial_reference": initial_reference,
        "initial_values_matched": True,
        "arms": arms,
        "differences_at_common_end": differences,
    }


def write_comparison_csv(path: Path, comparison: dict[str, object]) -> None:
    arms = comparison["arms"]  # type: ignore[assignment]
    off = arms["EMF_off"]  # type: ignore[index]
    on = arms["EMF_on"]  # type: ignore[index]
    initial_reference = comparison["initial_reference"]  # type: ignore[assignment]

    fieldnames = (
        "metric",
        "emf_off_initial_reference",
        "emf_off_at_common_end",
        "emf_on_at_common_end",
        "on_minus_off",
        "relative_on_minus_off",
        "emf_off_t50_Myr",
        "emf_on_t50_Myr",
        "on_minus_off_t50_Myr",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for metric in DISRUPTION_METRICS:
            off_t50 = off["t50_Myr"][metric]  # type: ignore[index]
            on_t50 = on["t50_Myr"][metric]  # type: ignore[index]
            writer.writerow(
                {
                    "metric": metric,
                    "emf_off_initial_reference": initial_reference[metric],  # type: ignore[index]
                    "emf_off_at_common_end": off["values_at_common_end"][metric],  # type: ignore[index]
                    "emf_on_at_common_end": on["values_at_common_end"][metric],  # type: ignore[index]
                    "on_minus_off": comparison["differences_at_common_end"][metric]["on_minus_off"],  # type: ignore[index]
                    "relative_on_minus_off": comparison["differences_at_common_end"][metric]["relative_on_minus_off"],  # type: ignore[index]
                    "emf_off_t50_Myr": off_t50,
                    "emf_on_t50_Myr": on_t50,
                    "on_minus_off_t50_Myr": None if off_t50 is None or on_t50 is None else on_t50 - off_t50,
                }
            )


def format_number(value: float | None, *, percent: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if percent:
        return f"{100.0 * value:+.2f}%"
    return f"{value:.6g}"


def print_comparison(comparison: dict[str, object]) -> None:
    arms = comparison["arms"]  # type: ignore[assignment]
    off = arms["EMF_off"]  # type: ignore[index]
    on = arms["EMF_on"]  # type: ignore[index]
    differences = comparison["differences_at_common_end"]  # type: ignore[assignment]

    print(f"\nComparison at common time {comparison['common_end_time_Myr']:.6g} Myr:")
    print(f"{'metric':31} {'EMF=0':>13} {'EMF=1':>13} {'relative delta':>15} {'t50 off [Myr]':>15} {'t50 on [Myr]':>14}")
    for metric in DISRUPTION_METRICS:
        print(
            f"{metric:31} "
            f"{format_number(off['values_at_common_end'][metric]):>13} "
            f"{format_number(on['values_at_common_end'][metric]):>13} "
            f"{format_number(differences[metric]['relative_on_minus_off'], percent=True):>15} "
            f"{format_number(off['t50_Myr'][metric]):>15} "
            f"{format_number(on['t50_Myr'][metric]):>14}"
        )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def default_output_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "sims" / f"MolecularCloudDisruption_EMF_AB_{timestamp}"


def parser_for(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("3d", "3d-debug", "3d-hip", "3d-cuda"), default="3d", help="Quokka build preset (default: %(default)s)")
    parser.add_argument("--resolution", type=validate_resolution, default=64, help="cells per box side; must be a multiple of 32 (default: %(default)s)")
    parser.add_argument("--input", type=Path, default=root / "inputs" / f"{PROBLEM}.toml", help="base input file")
    parser.add_argument("--particles-file", type=Path, default=root / "inputs" / f"{PROBLEM}_particles.txt", help="shared stellar-particle input")
    parser.add_argument(
        "--cooling-table",
        type=Path,
        default=root / "extern" / "cooling" / "CloudyData_UVB=HM2012_shielded_resampled_noPE.h5",
        help="shared cooling table",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(root), help="new directory for both arms and reports")
    parser.add_argument("--ranks", type=validate_positive_int, default=1, help="MPI ranks per arm (default: %(default)s)")
    parser.add_argument("--build-jobs", type=validate_positive_int, default=8, help="parallel build jobs (default: %(default)s)")
    parser.add_argument("--skip-build", action="store_true", help="use an existing MolecularCloudDisruption executable")
    parser.add_argument(
        "--override",
        action="append",
        type=parse_override,
        default=[],
        metavar="KEY=VALUE",
        help="extra runtime override applied identically to both arms; repeat as needed",
    )
    return parser


def resolve_existing(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


def main() -> None:
    root = repository_root()
    args = parser_for(root).parse_args()

    require_emf_implementation(root)
    input_file = resolve_existing(args.input, label="input file")
    particles_file = resolve_existing(args.particles_file, label="particle file")
    cooling_table = resolve_existing(args.cooling_table, label="cooling table")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists; choose a new path: {output_dir}")

    if not args.skip_build:
        build_command = [
            str(root / "scripts" / "bash" / "quokka"),
            "build",
            "-d",
            args.preset,
            PROBLEM,
            "-j",
            str(args.build_jobs),
        ]
        run_checked(build_command, cwd=root)

    executable = root / "build" / args.preset / "src" / "problems" / PROBLEM / PROBLEM
    if not executable.is_file():
        raise SystemExit(f"executable not found: {executable} (remove --skip-build or configure the {args.preset} preset)")
    if args.ranks > 1 and shutil.which("mpirun") is None:
        raise SystemExit("--ranks is greater than one, but mpirun was not found on PATH")

    max_grid_size = min(args.resolution, 128)
    common_overrides = [
        f"amr.n_cell={args.resolution} {args.resolution} {args.resolution}",
        f"amr.max_grid_size={max_grid_size}",
        "statistics_file=history.txt",
        "plotfile_prefix=plt",
        "checkpoint_prefix=chk",
        f'cooling.hdf5_data_file="{cooling_table}"',
        f"problem.stellar_particles_file={particles_file}",
        *args.override,
    ]

    commands: dict[str, list[str]] = {}
    for arm_name, emf_enabled in ARM_FLAGS.items():
        command = [str(executable), str(input_file), *common_overrides, f"particles.EMF_enabled={emf_enabled}"]
        if args.ranks > 1:
            command = ["mpirun", "-np", str(args.ranks), *command]
        commands[arm_name] = command

    output_dir.mkdir(parents=True)
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "problem": PROBLEM,
        "preset": args.preset,
        "resolution": args.resolution,
        "max_grid_size": max_grid_size,
        "input_file": str(input_file),
        "particles_file": str(particles_file),
        "cooling_table": str(cooling_table),
        "extra_overrides": args.override,
        "git": git_metadata(root),
        "arms": {name: {"EMF_enabled": ARM_FLAGS[name], "command": command, "status": "pending"} for name, command in commands.items()},
    }
    write_json(output_dir / "run_manifest.json", manifest)

    histories: dict[str, History] = {}
    elapsed_wall_s: dict[str, float] = {}
    for arm_name, command in commands.items():
        arm_dir = output_dir / arm_name
        arm_dir.mkdir()
        log_path = arm_dir / "run.log"
        print(f"\nRunning {arm_name} (particles.EMF_enabled={ARM_FLAGS[arm_name]}) in {arm_dir}", flush=True)
        start = time.monotonic()
        with log_path.open("w") as log:
            print(f"+ {shlex.join(command)}", file=log, flush=True)
            result = subprocess.run(command, cwd=arm_dir, stdout=log, stderr=subprocess.STDOUT)
        elapsed_wall_s[arm_name] = time.monotonic() - start
        arm_manifest = manifest["arms"][arm_name]  # type: ignore[index]
        arm_manifest["elapsed_wall_s"] = elapsed_wall_s[arm_name]  # type: ignore[index]
        arm_manifest["returncode"] = result.returncode  # type: ignore[index]
        arm_manifest["status"] = "completed" if result.returncode == 0 else "failed"  # type: ignore[index]
        write_json(output_dir / "run_manifest.json", manifest)
        if result.returncode != 0:
            raise SystemExit(f"{arm_name} failed with exit code {result.returncode}; see {log_path}")
        histories[arm_name] = parse_history(arm_dir / "history.txt")

    comparison = make_comparison(histories, elapsed_wall_s)
    comparison["resolution"] = args.resolution
    comparison["output_directory"] = str(output_dir)
    write_json(output_dir / "comparison.json", comparison)
    write_comparison_csv(output_dir / "comparison.csv", comparison)
    print_comparison(comparison)
    print(f"\nWrote A/B results to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}: {shlex.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
