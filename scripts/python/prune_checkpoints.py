#!/usr/bin/env python3
"""Keep only the checkpoints whose time is closest to multiples of a given interval.

Usage:
    prune_checkpoints.py <sim_dir> <interval> [--delete]

Example:
    prune_checkpoints.py . '3*Myr'          # dry run: print what would be kept/deleted
    prune_checkpoints.py . '3*Myr' --delete # actually delete

The simulation time is read from row 5 of <chk*>/Header. Units (yr, kyr, Myr, Gyr)
follow src/util/time_units.hpp (Julian year = 365.25 days).

The newest checkpoint and the one pointed to by the last_chk symlink are always kept,
so that the run stays restartable.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# Checkpoint folders are named chk followed by the step number, e.g. chk0000010.
CHK_PATTERN = re.compile(r"chk\d+")

# First unit name appearing in the interval expression; used as the display unit.
UNIT_PATTERN = re.compile(r"\b(Gyr|Myr|kyr|yr|s)\b")

# Time unit conversion factors to CGS seconds; see src/util/time_units.hpp
UNITS = {
    "s": 1.0,
    "yr": 3.15576e7,
    "kyr": 3.15576e10,
    "Myr": 3.15576e13,
    "Gyr": 3.15576e16,
}


def parse_time(expr):
    """Evaluate a time expression such as '3*Myr' or '2.5*Myr + 500*kyr' into seconds."""
    try:
        value = eval(expr, {"__builtins__": {}}, dict(UNITS))  # noqa: S307
    except Exception as exc:
        sys.exit(f"error: cannot parse time expression {expr!r}: {exc}")
    if not isinstance(value, (int, float)) or value <= 0.0:
        sys.exit(f"error: time expression {expr!r} must evaluate to a positive number")
    return float(value)


def read_checkpoint_time(chk_dir):
    """Return the simulation time (seconds) of level 0.

    Row 5 of the checkpoint Header is the t_new array, with one entry per level;
    see AMRSimulation::WriteCheckpointFile in src/simulation.hpp.
    """
    header = chk_dir / "Header"
    if not header.is_file():
        return None
    lines = header.read_text().splitlines()
    if len(lines) < 5:
        return None
    fields = lines[4].split()
    if not fields:
        return None
    try:
        return float(fields[0])
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_dir", help="simulation directory containing chk* folders")
    parser.add_argument("interval", help="time interval, e.g. '3*Myr'")
    parser.add_argument("--delete", action="store_true", help="actually delete instead of dry run")
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    if not sim_dir.is_dir():
        sys.exit(f"error: {sim_dir} is not a directory")

    interval = parse_time(args.interval)

    checkpoints = []
    for chk_dir in sorted(p for p in sim_dir.glob("chk*") if p.is_dir() and CHK_PATTERN.fullmatch(p.name)):
        time = read_checkpoint_time(chk_dir)
        if time is None:
            print(f"warning: skipping {chk_dir.name} (no readable time in Header)", file=sys.stderr)
            continue
        checkpoints.append((chk_dir, time))

    if not checkpoints:
        sys.exit(f"error: no checkpoints with a readable Header found in {sim_dir}")

    # Group checkpoints by the nearest multiple of the interval; keep the closest one per group.
    keep = {}
    for chk_dir, time in checkpoints:
        index = round(time / interval)
        best = keep.get(index)
        if best is None or abs(time - index * interval) < abs(best[1] - index * interval):
            keep[index] = (chk_dir, time)
    reasons = {chk_dir: f"~ {index} x {args.interval}" for index, (chk_dir, _) in keep.items()}

    # Always protect the newest checkpoint and whatever the last_chk symlink points to.
    newest = max(checkpoints, key=lambda item: item[1])[0]
    reasons[newest] = f"{reasons[newest]}, newest" if newest in reasons else "newest"
    last_chk = sim_dir / "last_chk"
    if last_chk.is_symlink():
        target = last_chk.resolve()
        for chk_dir, _ in checkpoints:
            if chk_dir.resolve() == target:
                reasons[chk_dir] = f"{reasons[chk_dir]}, last_chk" if chk_dir in reasons else "last_chk"

    match = UNIT_PATTERN.search(args.interval)
    unit = match.group(1) if match is not None else "s"
    print(f"interval = {args.interval} = {interval / UNITS[unit]:g} {unit}")
    n_delete = 0
    for chk_dir, time in checkpoints:
        if chk_dir in reasons:
            print(f"  {chk_dir.name}  {time / UNITS[unit]:12.4f} {unit}  [+] keep   ({reasons[chk_dir]})")
        else:
            n_delete += 1
            print(f"  {chk_dir.name}  {time / UNITS[unit]:12.4f} {unit}  [-] delete")

    print(f"\n{len(reasons)} to keep, {n_delete} to delete")

    if not args.delete:
        print("dry run: nothing deleted (pass --delete to remove the folders marked [-])")
        return

    print()
    for chk_dir, _ in checkpoints:
        if chk_dir not in reasons:
            print(f"deleting {chk_dir} ...", end=" ", flush=True)
            shutil.rmtree(chk_dir)
            print("done")
    print(f"deleted {n_delete} checkpoint folder(s)")


if __name__ == "__main__":
    main()
