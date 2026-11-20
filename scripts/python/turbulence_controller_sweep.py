#!/usr/bin/env python3
"""Run paired real-CFD turbulence startup comparisons; no surrogate dynamics."""

import argparse
import csv
import json
import math
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--seeds", default="140281,42,140282,271828")
    parser.add_argument("--amplitudes", default="1.5")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--cadence", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--target", type=float, default=4.57)
    parser.add_argument("--initial", type=float, default=0.0,
                        help="Initial transverse shear dispersion / target")
    parser.add_argument("--methods", default="legacy,proportional")
    args = parser.parse_args()
    executable = args.executable.resolve()
    root = pathlib.Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tau = 1.0 / (4.649213465060362 * args.target)
    results = []
    for seed in map(int, args.seeds.split(",")):
        for amplitude in map(float, args.amplitudes.split(",")):
            for method in args.methods.split(","):
                if method not in ("legacy", "proportional"):
                    parser.error("methods must be legacy and/or proportional")
                directory = output / f"{method}_seed{seed}_amplitude{amplitude}"
                # Require a fresh run directory; never reuse stale diagnostics.
                directory.mkdir()
                command = [str(executable), str(root / "inputs/TurbulenceStartup.toml"),
                           f"amr.n_cell={args.resolution} {args.resolution} {args.resolution}",
                           f"amr.blocking_factor={args.resolution}",
                           f"stop_time={6 * tau}", f"cfl={args.cfl}",
                           f"turbulence.target_vdisp={args.target}",
                           f"turbulence.ampl_factor={amplitude}",
                           f"turbulence.random_seed={seed}",
                           f"turbulence.nsteps_per_t_turb={args.cadence}",
                           f"turbulence.ampl_auto_adjust_method={method}",
                           f"problem.initial_vdisp={args.initial * args.target}",
                           "problem.output_dispersion=1", "problem.check_startup=0"]
                with (directory / "run.log").open("w") as log:
                    run = subprocess.run(command, cwd=directory, stdout=log,
                                         stderr=subprocess.STDOUT, check=False)
                data_path = directory / "dispersion.csv"
                if not data_path.exists():
                    raise RuntimeError(f"No diagnostics from {directory}; exit {run.returncode}")
                with data_path.open() as data:
                    rows = [(float(t) / tau, float(v) / args.target)
                            for t, v in csv.reader(data)]
                if not rows or not all(math.isfinite(t) and math.isfinite(v) for t, v in rows):
                    raise RuntimeError(f"Invalid diagnostics from {directory}")
                late = []
                previous = 0.0
                for t, v in rows:
                    weight = max(0.0, t - max(previous, 2.0))
                    late.append((weight, v))
                    previous = t
                duration = sum(w for w, _ in late)
                mean = sum(w * v for w, v in late) / duration
                rms = math.sqrt(sum(w * (v - mean) ** 2 for w, v in late) / duration)
                result = dict(method=method, seed=seed, amplitude=amplitude,
                              resolution=args.resolution, cadence=args.cadence, cfl=args.cfl,
                              target=args.target, initial=args.initial, exit=run.returncode,
                              end=rows[-1][0], peak=max(v for _, v in rows),
                              t90=next((t for t, v in rows if v >= 0.9), None),
                              at_one=next((v for t, v in rows if t >= 1.0), None),
                              mean=mean, std=rms)
                (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                print(json.dumps(result), flush=True)
                (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
