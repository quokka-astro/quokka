#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_git(worktree: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_output(worktree: Path, args: list[str]) -> str:
    return run_git(worktree, args).stdout.strip()


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    if not slug:
        raise ValueError("description must contain at least one alphanumeric character")
    return slug


def ensure_quokka_worktree(path: Path) -> Path:
    worktree = path.resolve()
    try:
        root = Path(git_output(worktree, ["rev-parse", "--show-toplevel"])).resolve()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{worktree} is not inside a git worktree: {exc.stderr.strip()}") from exc

    if not (root / "src" / "QuokkaSimulation.hpp").exists() or not (root / "scripts" / "bash" / "quokka").exists():
        raise RuntimeError(f"{root} does not look like a Quokka worktree")
    return root


def write_text(path: Path, content: str, *, executable: bool = False) -> None:
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def copy_files(files: list[Path], destination: Path, worktree: Path) -> None:
    for source in files:
        source_path = source if source.is_absolute() else worktree / source
        if not source_path.exists():
            raise FileNotFoundError(f"{source_path} does not exist")
        if source_path.is_dir():
            target = destination / source_path.name
            shutil.copytree(source_path, target, dirs_exist_ok=True)
        else:
            target = destination / source_path.name
            shutil.copy2(source_path, target)


def copy_untracked_files(paths: list[str], destination: Path, worktree: Path) -> None:
    if not paths:
        return
    for path in paths:
        source = worktree / path
        if not source.is_file():
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def get_untracked_files(worktree: Path) -> list[str]:
    output = git_output(worktree, ["ls-files", "--others", "--exclude-standard"])
    return [line for line in output.splitlines() if line.strip()]


def find_base_commit(worktree: Path, base_ref: str) -> str:
    try:
        run_git(worktree, ["rev-parse", "--verify", base_ref])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"base ref {base_ref!r} was not found; fetch the official Quokka remote or pass --base-ref"
        ) from exc
    return git_output(worktree, ["merge-base", "HEAD", base_ref])


def build_readme(
    slug: str,
    worktree: Path,
    branch: str,
    commit: str,
    base_ref: str,
    base_commit: str,
    status: str,
    untracked: list[str],
) -> str:
    untracked_text = "\n".join(f"- `{path}`" for path in untracked) if untracked else "- None detected."
    status_text = status if status else "Clean worktree."
    return f"""# Quokka bug reproducer: {slug}

## Summary

Describe the bug in one or two paragraphs.

## Expected behavior

Describe what should have happened.

## Observed behavior

Describe what happened instead. Include whether the failure is deterministic.

## How to run

```bash
bash run.sh
```

## Source checkout used to create this reproducer

- Quokka worktree: `{worktree}`
- Branch: `{branch}`
- Commit: `{commit}`
- Patch base ref: `{base_ref}`
- Patch base commit: `{base_commit}`

## Platform

- Host or cluster:
- Operating system:
- Compiler and version:
- MPI implementation and version:
- GPU backend and hardware, if any:
- Build preset or CMake command:

## Reduction notes

- Smallest domain size tested:
- Smallest MPI rank count tested:
- First failing timestep:
- Does it fail every time?
- What was removed from the original production case?

## Included files

- `run.sh`: edit this to configure, build, and run the failing case.
- `input/`: complete input deck(s).
- `data/`: required tables and auxiliary input files.
- `patches/quokka.patch`: code modifications from `git diff {base_commit} --binary`.
- `output/failure.log`: full output from the failing run.
- `output/expected.txt`: short expected-result note.

## Worktree status at creation time

```text
{status_text}
```

## Untracked files copied separately

{untracked_text}
"""


def build_run_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Edit these values before uploading the reproducer.
QUOKKA_ROOT="${QUOKKA_ROOT:-${SCRIPT_DIR}/../quokka}"
PRESET="${PRESET:-3d}"
PROBLEM="${PROBLEM:-MyProblem}"
INPUT="${INPUT:-${SCRIPT_DIR}/input/MyProblem.toml}"
MPI_RANKS="${MPI_RANKS:-1}"

cd "${QUOKKA_ROOT}"

# Apply local code changes if this reproducer includes any.
# The patch was generated relative to the base commit recorded in README.md.
if [[ -s "${SCRIPT_DIR}/patches/quokka.patch" ]]; then
  git apply --check "${SCRIPT_DIR}/patches/quokka.patch"
  git apply "${SCRIPT_DIR}/patches/quokka.patch"
fi

# Restore untracked files that cannot be represented by git diff.
if [[ -d "${SCRIPT_DIR}/patches/untracked-files" ]]; then
  cp -R "${SCRIPT_DIR}/patches/untracked-files/." .
fi

./scripts/bash/quokka config -d "${PRESET}"
./scripts/bash/quokka build -d "${PRESET}" "${PROBLEM}"

mpirun -np "${MPI_RANKS}" "./build/${PRESET}/src/problems/${PROBLEM}/${PROBLEM}" "${INPUT}"
"""


def create_reproducer(args: argparse.Namespace) -> Path:
    worktree = ensure_quokka_worktree(args.worktree)
    slug = sanitize_slug(args.description)
    repro_dir = args.output_dir.resolve() / f"quokka-reproducer-{slug}"
    untracked_files_dir = repro_dir / "patches" / "untracked-files"

    if repro_dir.exists():
        if not args.force:
            raise FileExistsError(f"{repro_dir} already exists; pass --force to overwrite it")
        shutil.rmtree(repro_dir)

    branch = git_output(worktree, ["branch", "--show-current"]) or "(detached HEAD)"
    commit = git_output(worktree, ["rev-parse", "HEAD"])
    base_commit = find_base_commit(worktree, args.base_ref)
    status = git_output(worktree, ["status", "--short"])
    untracked = get_untracked_files(worktree)

    input_dir = repro_dir / "input"
    data_dir = repro_dir / "data"
    patches_dir = repro_dir / "patches"
    output_dir = repro_dir / "output"
    for directory in (input_dir, data_dir, patches_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    diff = run_git(worktree, ["diff", base_commit, "--binary"]).stdout
    write_text(patches_dir / "quokka.patch", diff)
    write_text(
        patches_dir / "README.md",
        f"quokka.patch was generated with: git diff {base_commit} --binary\n"
        f"Base ref used to find that commit: {args.base_ref}\n"
        "Place additional patch notes here if the diff needs explanation.\n",
    )
    copy_untracked_files(untracked, untracked_files_dir, worktree)

    write_text(repro_dir / "README.md", build_readme(slug, worktree, branch, commit, args.base_ref, base_commit, status, untracked))
    write_text(repro_dir / "run.sh", build_run_script(), executable=True)
    write_text(input_dir / "README.md", "Put the complete Quokka input deck here, including every required parameter.\n")
    write_text(data_dir / "README.md", "Put required data tables and auxiliary input files here.\n")
    write_text(output_dir / "failure.log", "Paste the full failing terminal output here.\n")
    write_text(output_dir / "expected.txt", "Describe the expected result here.\n")

    copy_files(args.input, input_dir, worktree)
    copy_files(args.data, data_dir, worktree)

    return repro_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a skeleton self-contained Quokka bug reproducer directory.",
    )
    parser.add_argument(
        "description",
        help="Short description used in the directory name, e.g. dtypefront-pil-crash",
    )
    parser.add_argument(
        "--worktree",
        type=Path,
        default=Path("."),
        help="Quokka worktree to capture a patch from (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the reproducer directory is created (default: current directory)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Input file or directory to copy into input/; may be passed multiple times",
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        default=[],
        help="Data file or directory to copy into data/; may be passed multiple times",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/development",
        help="Official Quokka development ref used to find the patch base (default: origin/development)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing reproducer directory with the same name",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        repro_dir = create_reproducer(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"created {repro_dir}")
    print("edit README.md, run.sh, input/, data/, and output/ before uploading")
    print(f"after reviewing the contents: tar -czf {repro_dir.name}.tar.gz {repro_dir.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
