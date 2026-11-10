#!/usr/bin/env python3
"""Print a bitwise fingerprint of AMReX plotfiles.

For each plotfile this prints one SHA-256 line per file (everything except
metadata.yaml, which records the git hash, and job_info, which records the
wall-clock date) and one line per level and variable with the SHA-256 of that
variable's raw cell data plus its min, max and sum to 17 significant digits.
The per-variable lines carry the same information as the Cell_D hashes; they
are there so that a diff of two fingerprints names the variable that changed
and shows roughly by how much.

Usage: plotfile_hashes.py PLOTFILE [PLOTFILE ...]

Only the standard library and numpy are used.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

EXCLUDED_FILES = {"metadata.yaml", "job_info"}

# e.g. FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((0,0,0) (7,7,7) (0,0,0)) 13
FAB_HEADER = re.compile(
    r"FAB \(\((\d+), \([^)]*\)\),\((\d+), \(([\d ]+)\)\)\)"
    r"\(\(([-\d,]+)\) \(([-\d,]+)\) \([-\d,]+\)\) (\d+)"
)
FAB_ON_DISK = re.compile(r"FabOnDisk: (\S+) (\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fabs(level_dir: Path):
    """Yield (dtype, ncomp, raw bytes) for every FAB of a level, in Cell_H order."""
    for line in (level_dir / "Cell_H").read_text().splitlines():
        match = FAB_ON_DISK.match(line.strip())
        if match is None:
            continue
        with (level_dir / match.group(1)).open("rb") as stream:
            stream.seek(int(match.group(2)))
            header = stream.readline().decode("ascii").strip()
            fab = FAB_HEADER.match(header)
            if fab is None:
                raise ValueError(f"unrecognised FAB header in {level_dir / match.group(1)}: {header}")
            nbytes = int(fab.group(2))
            order = [int(b) for b in fab.group(3).split()]
            endian = "<" if order[0] == nbytes else ">"
            lo = [int(x) for x in fab.group(4).split(",")]
            hi = [int(x) for x in fab.group(5).split(",")]
            ncomp = int(fab.group(6))
            npts = int(np.prod([h - l + 1 for l, h in zip(lo, hi)]))
            raw = stream.read(npts * ncomp * nbytes)
            if len(raw) != npts * ncomp * nbytes:
                raise ValueError(f"truncated FAB in {level_dir / match.group(1)}")
            yield np.dtype(f"{endian}f{nbytes}"), ncomp, raw


def fingerprint(plotfile: Path) -> list[str]:
    lines = [f"plotfile {plotfile.name}"]
    for path in sorted(p for p in plotfile.rglob("*") if p.is_file()):
        if path.name not in EXCLUDED_FILES:
            lines.append(f"file {path.relative_to(plotfile).as_posix()} {sha256_file(path)}")

    header = (plotfile / "Header").read_text().splitlines()
    nvars = int(header[1])
    varnames = header[2 : 2 + nvars]
    for level_dir in sorted(plotfile.glob("Level_*"), key=lambda p: int(p.name.split("_")[1])):
        if not (level_dir / "Cell_H").is_file():
            continue
        digests = [hashlib.sha256() for _ in varnames]
        values: list[list[np.ndarray]] = [[] for _ in varnames]
        for dtype, ncomp, raw in read_fabs(level_dir):
            if ncomp != nvars:
                raise ValueError(f"{level_dir}: FAB has {ncomp} components, Header lists {nvars}")
            chunk = len(raw) // ncomp
            for comp in range(ncomp):
                comp_raw = raw[comp * chunk : (comp + 1) * chunk]
                digests[comp].update(comp_raw)
                values[comp].append(np.frombuffer(comp_raw, dtype=dtype))
        for comp, name in enumerate(varnames):
            data = np.concatenate(values[comp]) if values[comp] else np.zeros(0)
            stats = (
                f"min={data.min():.17g} max={data.max():.17g} sum={data.sum():.17g}" if data.size else "empty"
            )
            lines.append(f"var {level_dir.name} {name} {digests[comp].hexdigest()} {stats}")
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        print("\n".join(fingerprint(Path(arg))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
