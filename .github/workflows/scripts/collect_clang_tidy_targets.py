#!/usr/bin/env python3
"""Select translation units to run clang-tidy on for a PR diff."""

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set


SOURCE_EXTS = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh"}
HEADER_EXTS = {".h", ".hh", ".hpp", ".hxx"}


def _read_changed_files(path: pathlib.Path, repo_root: pathlib.Path) -> List[pathlib.Path]:
    changed: List[pathlib.Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        changed.append((repo_root / stripped).resolve())
    return changed


def _load_compile_commands(path: pathlib.Path, repo_root: pathlib.Path) -> Set[pathlib.Path]:
    commands = json.loads(path.read_text(encoding="utf-8"))
    tus: Set[pathlib.Path] = set()
    for entry in commands:
        tu_path = pathlib.Path(entry["file"])
        if not tu_path.is_absolute():
            tu_path = (pathlib.Path(entry["directory"]) / tu_path).resolve()
        try:
            tu_path.relative_to(repo_root)
        except ValueError:
            # Skip entries outside the repository.
            continue
        tus.add(tu_path)
    return tus


def _build_include_index(tus: Iterable[pathlib.Path], repo_root: pathlib.Path) -> Dict[str, Set[pathlib.Path]]:
    """Map include strings (full and basenames) to translation units that reference them."""
    include_index: Dict[str, Set[pathlib.Path]] = defaultdict(set)
    include_re = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)

    for tu in tus:
        try:
            content = tu.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in include_re.finditer(content):
            include_target = match.group(1)
            include_index[include_target].add(tu)
            include_index[pathlib.Path(include_target).name].add(tu)
        try:
            rel_path = tu.relative_to(repo_root).as_posix()
            include_index[rel_path].add(tu)
        except ValueError:
            # TU is not relative to repo root; skipping.
            pass
    return include_index


def _candidates_for_header(
    header: pathlib.Path, include_index: Dict[str, Set[pathlib.Path]], available_tus: Set[pathlib.Path], repo_root: pathlib.Path
) -> Set[pathlib.Path]:
    candidates: Set[pathlib.Path] = set()

    rel_posix = header.relative_to(repo_root).as_posix()
    include_keys = {rel_posix, header.name}

    if rel_posix.startswith("src/"):
        include_keys.add(rel_posix[len("src/") :])

    for key in include_keys:
        candidates.update(include_index.get(key, set()))

    # Heuristic: match sources with the same stem as the header.
    stem = header.with_suffix("").name
    for ext in SOURCE_EXTS:
        sibling = header.with_suffix(ext)
        if sibling in available_tus:
            candidates.add(sibling)
        if rel_posix.startswith("src/"):
            src_less = repo_root / rel_posix[len("src/") :]
            sibling_src_less = src_less.with_suffix(ext)
            if sibling_src_less in available_tus:
                candidates.add(sibling_src_less)
        sibling_same_dir = header.parent / f"{stem}{ext}"
        if sibling_same_dir in available_tus:
            candidates.add(sibling_same_dir)

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve clang-tidy targets from a changed-file list.")
    parser.add_argument("--changed-files", required=True, help="Path to file containing changed paths (relative to repo root).")
    parser.add_argument("--compile-commands", required=True, help="Path to compile_commands.json.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root.")
    parser.add_argument("--output", required=True, help="Output file with translation units for clang-tidy.")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    changed_file_path = pathlib.Path(args.changed_files)
    compile_commands_path = pathlib.Path(args.compile_commands)
    output_path = pathlib.Path(args.output)

    changed_files = _read_changed_files(changed_file_path, repo_root)
    available_tus = _load_compile_commands(compile_commands_path, repo_root)
    include_index = _build_include_index(available_tus, repo_root)

    targets: Set[pathlib.Path] = set()
    for path in changed_files:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix in SOURCE_EXTS:
            if path in available_tus:
                targets.add(path)
        elif suffix in HEADER_EXTS:
            targets.update(_candidates_for_header(path, include_index, available_tus, repo_root))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for tu in sorted(targets):
            fout.write(f"{tu}\n")


if __name__ == "__main__":
    main()
