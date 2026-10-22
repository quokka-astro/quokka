#!/usr/bin/env python3
"""Check that every CUDA translation unit passes the nvcc front end, without building.

Reads compile_commands.json from a configured CUDA build directory and re-runs each
unique nvcc compile with ``-ptx -o /dev/null`` in place of ``-c -o <object>``. nvcc then
runs only cudafe++ and cicc: no ptxas, host compile, device link or link. This catches
code that nvcc rejects while taking a fraction of the time of a full build.

Compiles that differ only in their output paths (the same source rebuilt for each
problem target with identical flags) are run once.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

DROP_WITH_VALUE = {"-o", "-MT", "-MF"}
DROP = {"-c", "-MD"}


def ptx_command(entry: dict) -> tuple[tuple, list[str]] | None:
    """The -ptx form of one nvcc compile and a key for its inputs and flags; None if not nvcc."""
    argv = shlex.split(entry["command"])
    if Path(argv[0]).name == "ccache":
        argv = argv[1:]
    if Path(argv[0]).name != "nvcc":
        return None
    command, key, skip, have_arch = [], [entry["file"]], False, False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg in DROP_WITH_VALUE:
            skip = True
            continue
        if arg in DROP:
            continue
        if arg.startswith("--generate-code="):
            # -ptx needs a single target; the first virtual architecture runs the same front end.
            if have_arch:
                continue
            have_arch = True
            arch = arg.split("arch=", 1)[1].split(",", 1)[0]
            arg = f"--generate-code=arch={arch},code={arch}"
        command.append(arg)
        if i and argv[i - 1] == "--options-file":
            # Per-target response files differ by path but are often identical in content.
            key.append((Path(entry["directory"]) / arg).read_text())
        elif arg != "--options-file":
            key.append(arg)
    return tuple(key), command + ["-ptx", "-o", os.devnull]


def run(item: tuple[str, list[str], str]) -> tuple[str, int, str, float]:
    directory, command, source = item
    started = time.monotonic()
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    return source, result.returncode, result.stdout, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path, help="configured build directory with compile_commands.json")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="parallel compiles (default: all CPUs)")
    args = parser.parse_args()

    entries = json.loads((args.build_dir / "compile_commands.json").read_text())
    unique = {}
    for entry in entries:
        ptx = ptx_command(entry)
        if ptx is not None:
            unique.setdefault(ptx[0], (entry["directory"], ptx[1], entry["file"]))
    items = list(unique.values())
    print(f"{len(items)} unique nvcc compiles (of {len(entries)} compile commands), {args.jobs} at a time", flush=True)

    started = time.monotonic()
    failures, killed, times = [], [], []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for source, code, output, seconds in pool.map(run, items):
            if code < 0:
                killed.append(source)
            elif code:
                failures.append((source, output))
            else:
                times.append((seconds, source))
    if killed:
        # Usually memory: retry those one at a time rather than report a false failure.
        print(f"{len(killed)} compiles were killed; retrying them serially", flush=True)
        for item in (i for i in items if i[2] in set(killed)):
            source, code, output, seconds = run(item)
            if code:
                failures.append((source, output))
            else:
                times.append((seconds, source))

    print(f"Front-end check finished in {time.monotonic() - started:.0f} s")
    for seconds, source in sorted(times, reverse=True)[:5]:
        print(f"  {seconds:6.1f} s  {source}")
    for source, output in failures:
        print(f"\n::error file={source}::nvcc front end failed\n{output}")
    if failures:
        print(f"\n{len(failures)} of {len(items)} compiles failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
