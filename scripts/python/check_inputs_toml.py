#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def main() -> int:
    inputs_dir = Path("inputs")
    toml_files = sorted(inputs_dir.glob("*.toml"))
    legacy_files = sorted(inputs_dir.glob("*.in"))

    if legacy_files:
        legacy_paths = ", ".join(path.as_posix() for path in legacy_files)
        print(f"legacy ParmParse inputs are still present: {legacy_paths}", file=sys.stderr)
        return 1

    if not toml_files:
        print("no TOML input files found in inputs/", file=sys.stderr)
        return 1

    for path in toml_files:
        try:
            tomllib.loads(path.read_text())
        except Exception as exc:
            print(f"{path.as_posix()}: {exc}", file=sys.stderr)
            return 1

    print(f"validated {len(toml_files)} TOML input files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
