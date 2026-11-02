#!/usr/bin/env python3
"""Run paired AoS/RTSoA executables and compare named TinyProfiler regions.

Run on an otherwise idle host. Each sample gets a new directory and retains its
output. This measures the selected problem; it does not establish that a small
regression case is representative of a particle-dominated production workload.
"""

import argparse
import hashlib
import json
import pathlib
import platform
import re
import statistics
import subprocess
import time


def inclusive_times(output):
    """Read the top-level inclusive table, excluding repeated named-region tables."""
    rows = {}
    active = False
    for line in output.splitlines():
        if line.startswith("BEGIN REGION"):
            break
        if "NCalls" in line and "Incl. Min" in line:
            active = True
            continue
        if active:
            match = re.fullmatch(
                r"(.+?)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+[\d.]+%", line
            )
            if match:
                name, calls, minimum, average, maximum = match.groups()
                rows[name.strip()] = {
                    "calls": int(calls),
                    "min_seconds": float(minimum),
                    "avg_seconds": float(average),
                    "max_seconds": float(maximum),
                }
            elif rows and not line.strip():
                break
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos", required=True, type=pathlib.Path)
    parser.add_argument("--rtsoa", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, type=pathlib.Path, help="Input/dependency tree shared by both variants")
    parser.add_argument("--input", required=True, help="File name under source/inputs")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="New directory for logs and summary.json")
    parser.add_argument("--backend", required=True, choices=["CPU", "CUDA", "HIP"])
    parser.add_argument("--profile", required=True, action="append", help="Exact TinyProfiler region name; repeatable")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("parameters", nargs="*", help="Identical runtime overrides for both variants")
    args = parser.parse_args()
    if args.repetitions < 3 or args.warmups < 0:
        parser.error("Use at least three measured repetitions and a nonnegative warmup count")
    executables = {name: getattr(args, name).resolve(strict=True) for name in ("aos", "rtsoa")}
    source = args.source.resolve(strict=True)
    input_file = (source / "inputs" / args.input).resolve(strict=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    parameters = args.parameters + [
        "tiny_profiler.enabled=1", "tiny_profiler.print_threshold=0", "tiny_profiler.output_file=stdout",
        "tiny_profiler.device_synchronize_around_region=" + ("0" if args.backend == "CPU" else "1"),
    ]
    report = {
        "host": platform.node(), "platform": platform.platform(), "backend": args.backend,
        "executables": {name: {"path": str(path), "sha256": sha256(path)} for name, path in executables.items()},
        "input": {"path": str(input_file), "sha256": sha256(input_file)},
        "parameters": parameters, "profiles": args.profile, "samples": [],
        "gpu_timing": "Synchronized profiler regions; synchronization can perturb overlap and throughput",
    }

    def save():
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for iteration in range(args.warmups + args.repetitions):
        for variant in (("aos", "rtsoa") if iteration % 2 == 0 else ("rtsoa", "aos")):
            directory = root / f"{iteration:02d}-{variant}"
            run = directory / "run"
            run.mkdir(parents=True)
            for name in ("inputs", "extern"):
                (directory / name).symlink_to(source / name, target_is_directory=True)
            command = [str(executables[variant]), str(input_file)] + parameters
            started = time.monotonic()
            with (directory / "output.log").open("w") as log:
                result = subprocess.run(command, cwd=run, stdout=log, stderr=subprocess.STDOUT)
            output = (directory / "output.log").read_text()
            detected_backend = next((backend for backend in ("CUDA", "HIP")
                                     if f"Initializing {backend}..." in output), "CPU")
            sample = {
                "variant": variant, "iteration": iteration, "warmup": iteration < args.warmups,
                "command": command, "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "detected_backend": detected_backend, "regions": inclusive_times(output),
            }
            report["samples"].append(sample)
            save()
            result.check_returncode()
            if detected_backend != args.backend:
                raise RuntimeError(f"Requested {args.backend}, but {directory} initialized {detected_backend}")
            missing = set(args.profile) - sample["regions"].keys()
            if missing:
                raise RuntimeError(f"Requested profiler regions absent in {directory}: {sorted(missing)}")
            print(f"{iteration:02d} {variant}: completed", flush=True)

    comparisons = {}
    invalid = []
    for name in args.profile:
        variants = {}
        call_counts = set()
        for variant in executables:
            regions = [sample["regions"][name] for sample in report["samples"]
                       if sample["variant"] == variant and not sample["warmup"]]
            call_counts.update(region["calls"] for region in regions)
            values = [region["max_seconds"] for region in regions]
            variants[variant] = {"seconds": values, "median_seconds": statistics.median(values),
                                 "min_seconds": min(values), "max_seconds": max(values)}
        if len(call_counts) != 1 or 0 in call_counts:
            comparisons[name] = {"variants": variants, "comparable": False,
                                 "reason": "Inconsistent or zero call counts", "call_counts": sorted(call_counts)}
            invalid.append(name)
            continue
        baseline = variants["aos"]["median_seconds"]
        if baseline <= 0:
            comparisons[name] = {"variants": variants, "comparable": False, "reason": "Insufficient timer resolution"}
            invalid.append(name)
            continue
        change = 100 * (variants["rtsoa"]["median_seconds"] / baseline - 1)
        comparisons[name] = {"variants": variants, "comparable": True, "calls": call_counts.pop(),
                             "change_percent": change, "requires_investigation": change > 5}
    report["comparisons"] = comparisons
    save()
    print(json.dumps(comparisons, indent=2))
    if invalid:
        raise RuntimeError(f"Unusable comparisons retained in summary.json: {invalid}")


if __name__ == "__main__":
    main()
