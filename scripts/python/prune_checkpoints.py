#!/usr/bin/env python3
"""Prune checkpoint folders, either by time interval or by hand.

Usage:
    prune_checkpoints.py <sim_dir> <interval> [--delete]
    prune_checkpoints.py <sim_dir> --manual

Example:
    prune_checkpoints.py . '3*Myr'          # dry run: print what would be kept/deleted
    prune_checkpoints.py . '3*Myr' --delete # actually delete
    prune_checkpoints.py . --manual         # pick the checkpoints to keep interactively

The simulation time is read from row 5 of <chk*>/Header. Units (yr, kyr, Myr, Gyr)
follow src/util/time_units.hpp (Julian year = 365.25 days).

The newest checkpoint and the one pointed to by the last_chk symlink are always kept,
so that the run stays restartable.
"""

import argparse
import curses
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


def collect_checkpoints(sim_dir):
    """Return [(path, time)] for every chk<digits> folder with a readable Header, sorted by name."""
    checkpoints = []
    for chk_dir in sorted(p for p in sim_dir.glob("chk*") if p.is_dir() and CHK_PATTERN.fullmatch(p.name)):
        time = read_checkpoint_time(chk_dir)
        if time is None:
            print(f"warning: skipping {chk_dir.name} (no readable time in Header)", file=sys.stderr)
            continue
        checkpoints.append((chk_dir, time))
    return checkpoints


def add_reason(reasons, chk_dir, text):
    """Record why chk_dir is kept, appending to any reason already recorded."""
    reasons[chk_dir] = f"{reasons[chk_dir]}, {text}" if chk_dir in reasons else text


def protected_checkpoints(sim_dir, checkpoints):
    """Return {path: reason} for checkpoints that must never be deleted."""
    reasons = {}
    add_reason(reasons, max(checkpoints, key=lambda item: item[1])[0], "newest")
    last_chk = sim_dir / "last_chk"
    if last_chk.is_symlink():
        target = last_chk.resolve()
        for chk_dir, _ in checkpoints:
            if chk_dir.resolve() == target:
                add_reason(reasons, chk_dir, "last_chk")
    return reasons


def select_by_interval(checkpoints, interval, expr):
    """Return {path: reason} for the checkpoint closest to each multiple of the interval."""
    keep = {}
    for chk_dir, time in checkpoints:
        index = round(time / interval)
        best = keep.get(index)
        if best is None or abs(time - index * interval) < abs(best[1] - index * interval):
            keep[index] = (chk_dir, time)
    return {chk_dir: f"~ {index} x {expr}" for index, (chk_dir, _) in keep.items()}


def _selector(stdscr, checkpoints, protected, unit):
    """curses loop for manual mode; returns the set of selected paths, or None if aborted."""
    curses.curs_set(0)
    selected = {chk_dir for chk_dir, _ in checkpoints}
    row = 0
    top = 0
    message = ""
    while True:
        height, width = stdscr.getmaxyx()
        view = max(1, height - 4)
        top = min(top, row)
        top = max(top, row - view + 1)
        stdscr.erase()
        stdscr.addnstr(0, 0, "up/down: move   space: toggle   enter: confirm   q: abort", width - 1)
        stdscr.addnstr(1, 0, f"[x] keep, [ ] delete -- {len(selected)} of {len(checkpoints)} kept.  {message}", width - 1)
        for offset in range(min(view, len(checkpoints) - top)):
            chk_dir, time = checkpoints[top + offset]
            mark = "x" if chk_dir in selected else " "
            lock = f"  ({protected[chk_dir]}, locked)" if chk_dir in protected else ""
            line = f" [{mark}] {chk_dir.name}  {time / UNITS[unit]:12.4f} {unit}{lock}"
            attr = curses.A_REVERSE if top + offset == row else curses.A_NORMAL
            stdscr.addnstr(2 + offset, 0, line.ljust(width - 1), width - 1, attr)
        stdscr.refresh()

        key = stdscr.getch()
        message = ""
        if key in (curses.KEY_UP, ord("k")):
            row = max(0, row - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            row = min(len(checkpoints) - 1, row + 1)
        elif key == ord(" "):
            chk_dir = checkpoints[row][0]
            if chk_dir in protected:
                message = f"{chk_dir.name} is protected ({protected[chk_dir]})."
            elif chk_dir in selected:
                selected.remove(chk_dir)
            else:
                selected.add(chk_dir)
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return selected
        elif key == ord("q"):
            return None


def select_manually(checkpoints, protected, unit):
    """Run the interactive picker; returns {path: reason} for the checkpoints to keep."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        sys.exit("error: --manual requires an interactive terminal")
    selected = curses.wrapper(_selector, checkpoints, protected, unit)
    if selected is None:
        sys.exit("aborted: nothing deleted")
    return {chk_dir: "selected" for chk_dir in selected}


def confirm_deletion(n_delete):
    """Require the user to type an exact phrase before deleting."""
    phrase = f"delete {n_delete} checkpoint" + ("s" if n_delete != 1 else "")
    answer = input(f'\ntype "{phrase}" to confirm: ')
    if answer.strip() != phrase:
        sys.exit("aborted: nothing deleted")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_dir", help="simulation directory containing chk* folders")
    parser.add_argument("interval", nargs="?", help="time interval, e.g. '3*Myr'")
    parser.add_argument("--manual", action="store_true", help="pick the checkpoints to keep interactively")
    parser.add_argument("--delete", action="store_true", help="actually delete instead of dry run (interval mode only)")
    args = parser.parse_args()

    if args.manual == (args.interval is not None):
        parser.error("give either an interval or --manual, not both")

    sim_dir = Path(args.sim_dir)
    if not sim_dir.is_dir():
        sys.exit(f"error: {sim_dir} is not a directory")

    checkpoints = collect_checkpoints(sim_dir)
    if not checkpoints:
        sys.exit(f"error: no checkpoints with a readable Header found in {sim_dir}")

    protected = protected_checkpoints(sim_dir, checkpoints)

    if args.manual:
        unit = "Myr"
        reasons = select_manually(checkpoints, protected, unit)
    else:
        interval = parse_time(args.interval)
        match = UNIT_PATTERN.search(args.interval)
        unit = match.group(1) if match is not None else "s"
        reasons = select_by_interval(checkpoints, interval, args.interval)
        print(f"interval = {args.interval} = {interval / UNITS[unit]:g} {unit}")

    for chk_dir, reason in protected.items():
        add_reason(reasons, chk_dir, reason)

    n_delete = 0
    for chk_dir, time in checkpoints:
        if chk_dir in reasons:
            print(f"  {chk_dir.name}  {time / UNITS[unit]:12.4f} {unit}  [+] keep   ({reasons[chk_dir]})")
        else:
            n_delete += 1
            print(f"  {chk_dir.name}  {time / UNITS[unit]:12.4f} {unit}  [-] delete")

    print(f"\n{len(reasons)} to keep, {n_delete} to delete")

    if n_delete == 0:
        print("nothing to delete")
        return

    if args.manual:
        confirm_deletion(n_delete)
    elif not args.delete:
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
