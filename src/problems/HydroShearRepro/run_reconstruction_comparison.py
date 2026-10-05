#!/usr/bin/env python3
"""Run the 2D, dual-energy shear reproducer and its reconstruction controls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_history_time(path):
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    return float(lines[-1].split()[0]) if lines else None


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=root / "build/2d/src/problems/HydroShearRepro/HydroShearRepro")
    parser.add_argument("--input", type=Path, default=root / "inputs/HydroShearReproFailure.toml")
    parser.add_argument("--output", type=Path, default=root / "build/2d/shear-reconstruction-comparison")
    parser.add_argument("--stop-time", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="Maximum wall seconds per run")
    parser.add_argument("--expect-original-failure", action="store_true", help="Check the historical outcome with an unmodified executable")
    args = parser.parse_args()
    exe, input_file, output = args.exe.resolve(), args.input.resolve(), args.output.resolve()
    if not exe.is_file() or not input_file.is_file():
        parser.error("The compiled HydroShearRepro executable and input file must exist.")
    if not math.isfinite(args.stop_time) or args.stop_time <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--stop-time and --timeout must be positive finite numbers.")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "executable": str(exe),
        "executable_sha256": file_hash(exe),
        "input": str(input_file),
        "input_sha256": file_hash(input_file),
        "dimension": 2,
        "dual_energy": True,
        "stop_time": args.stop_time,
        "runs": [],
    }
    for viscosity, cfl in ((0.0, 0.3), (0.0, 0.15), (0.1, 0.3)):
        for order, reconstruction in ((3, "ppm"), (5, "xppm")):
            name = f"{reconstruction}-av{viscosity:g}-cfl{cfl:g}"
            run_dir = output / name
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(exe), str(input_file),
                "amr.n_cell=32 32",  # The driver also rejects builds with AMREX_SPACEDIM != 2.
                "amr.max_level=0", "amr.v=1",
                f"hydro.reconstruction_order={order}",
                "hydro.use_dual_energy=1",
                "hydro.rk_integrator_order=2",
                f"hydro.artificial_viscosity_coefficient={viscosity}",
                f"cfl={cfl}", f"stop_time={args.stop_time}",
                "shear.history_file=history.txt",
            ]
            (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            # Remove only our own previous history so a failed startup cannot reuse stale diagnostics.
            history = run_dir / "history.txt"
            history.unlink(missing_ok=True)
            start = time.monotonic()
            timed_out = False
            with (run_dir / "run.log").open("w") as log_file:
                try:
                    completed = subprocess.run(command, cwd=run_dir, stdout=log_file, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    exit_code = None
                    timed_out = True
            elapsed = time.monotonic() - start
            log_text = (run_dir / "run.log").read_text(errors="replace")
            retry_exhaustion = "Hydro update exceeded max_retries" in log_text
            reached_time = final_history_time(history)
            expected_failure = args.expect_original_failure and order == 5 and viscosity == 0.0 and cfl == 0.3
            if expected_failure:
                matched = not timed_out and exit_code not in (None, 0) and retry_exhaustion
            else:
                matched = (
                    not timed_out and exit_code == 0 and not retry_exhaustion
                    and reached_time is not None and math.isclose(reached_time, args.stop_time, rel_tol=1e-12, abs_tol=1e-12)
                )
            result = {
                "name": name, "command": command, "directory": str(run_dir),
                "reconstruction": reconstruction, "artificial_viscosity": viscosity, "cfl": cfl,
                "expected_outcome": "retry_exhaustion" if expected_failure else "completed",
                "exit_code": exit_code, "timed_out": timed_out, "elapsed_seconds": elapsed,
                "final_history_time": reached_time, "retry_exhaustion": retry_exhaustion,
                "retry_messages": log_text.count("Re-trying hydro advance"), "matched_expectation": matched,
            }
            report["runs"].append(result)
            (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            actual = "timeout" if timed_out else "retry_exhaustion" if retry_exhaustion else f"exit={exit_code}, t={reached_time}"
            print(f"{'PASS' if matched else 'FAIL'} {name}: {actual}", flush=True)
    all_matched = all(run["matched_expectation"] for run in report["runs"])
    print(f"Results: {output / 'results.json'}")
    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
