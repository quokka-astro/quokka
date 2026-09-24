#!/usr/bin/env python3
"""Require a Markdown parameter-table row for every registered ParmParse option."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ENTRY_RE = re.compile(r'^\s*X\("([^"]*)", "([^"]*)",', re.MULTILINE)
CALL_RE = re.compile(r"^\s*X\(", re.MULTILINE)


def registered_options(registry: Path) -> set[str]:
    source = registry.read_text(encoding="utf-8")
    entries = ENTRY_RE.findall(source)
    if not entries or len(entries) != len(CALL_RE.findall(source)):
        raise ValueError(f"Could not parse every option in {registry}")
    names = {f"{'<diagnostic>' if prefix == '*' else prefix}.{name}" if prefix else name for prefix, name in entries}
    if len(names) != len(entries):
        raise ValueError(f"Duplicate ParmParse options in {registry}")
    return names


def documented_options(docs_root: Path) -> set[str]:
    names: set[str] = set()
    for page in docs_root.rglob("*.md"):
        description_column: int | None = None
        for line in page.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|"):
                description_column = None
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] == "Parameter Name" and "Description" in cells:
                description_column = cells.index("Description")
                continue
            if description_column is None or len(cells) <= description_column:
                continue
            description = cells[description_column]
            if description and not description.startswith(("Option read in ", "See the reading code")):
                names.add(cells[0].strip("`"))
    return names


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=repo_root / "src/util/ParmParseOptionRegistry.hpp")
    parser.add_argument("--docs-root", type=Path, default=repo_root / "docs/markdown")
    args = parser.parse_args()

    missing = sorted(registered_options(args.registry) - documented_options(args.docs_root))
    if missing:
        print("Registered ParmParse options missing from Markdown parameter tables:")
        for name in missing:
            print(f"  - {name}")
        return 1
    print("Every registered ParmParse option has a Markdown parameter-table row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
