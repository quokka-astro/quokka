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


REPRODUCER_PREFIX = "quokka-reproducer-"


def get_untracked_files(worktree: Path, output_dir: Path) -> list[str]:
    """Return untracked, non-ignored files, skipping earlier reproducers and the output directory."""
    output = git_output(worktree, ["ls-files", "--others", "--exclude-standard"])
    files = []
    for line in output.splitlines():
        if not line.strip():
            continue
        path = Path(line)
        if any(part.startswith(REPRODUCER_PREFIX) for part in path.parts):
            continue
        if (worktree / path).resolve().is_relative_to(output_dir):
            continue
        files.append(line)
    return files


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
- Patch base commit: `{base_commit}` (merge-base of `HEAD` and `{base_ref}`)

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
- `patches/quokka.patch`: code modifications from `git diff --submodule=diff --binary {base_commit}`.
- `output/failure.log`: full output from the failing run.

`run.sh` creates a separate Quokka worktree in `quokka-src/` at the patch base commit,
syncs its submodules, applies the patch, and copies the untracked files. This happens
only on the first run, so later runs keep any edits made in `quokka-src/`.
The executable runs from this directory, so input decks can refer to `data/...`.

## Worktree status at creation time

```text
{status_text}
```

## Untracked files copied separately

{untracked_text}
"""


def build_run_script(base_ref: str, base_commit: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"

# The patch in patches/quokka.patch was generated relative to this commit.
PATCH_BASE_REF="{base_ref}"
PATCH_BASE_COMMIT="{base_commit}"

# Edit these values before uploading the reproducer.
PRESET="${{PRESET:-3d}}"
PROBLEM="${{PROBLEM:-MyProblem}}"
INPUT="${{INPUT:-${{SCRIPT_DIR}}/input/MyProblem.toml}}"
MPI_RANKS="${{MPI_RANKS:-1}}"

# Quokka clone that provides the git objects. By default, this is the checkout
# that contains this directory (unpack the reproducer in its tests/ directory).
QUOKKA_ROOT="${{QUOKKA_ROOT:-$(git -C "${{SCRIPT_DIR}}" rev-parse --show-toplevel 2>/dev/null || true)}}"
if [[ -z "${{QUOKKA_ROOT}}" ]]; then
  echo "error: unpack this reproducer inside a Quokka checkout (e.g. its tests/ directory) or set QUOKKA_ROOT" >&2
  exit 1
fi

# The reproducer builds in its own worktree, so the checkout in QUOKKA_ROOT is not modified.
SRC_DIR="${{SCRIPT_DIR}}/quokka-src"
SETUP_MARKER="${{SRC_DIR}}/.reproducer-setup-done"

echo "Patch base ref: ${{PATCH_BASE_REF}}"
echo "Patch base commit: ${{PATCH_BASE_COMMIT}}"
echo "Quokka clone: ${{QUOKKA_ROOT}}"
echo "Reproducer worktree: ${{SRC_DIR}}"

if [[ ! -f "${{SETUP_MARKER}}" ]]; then
  if ! git -C "${{QUOKKA_ROOT}}" cat-file -e "${{PATCH_BASE_COMMIT}}^{{commit}}" 2>/dev/null; then
    echo "error: commit ${{PATCH_BASE_COMMIT}} is not in ${{QUOKKA_ROOT}}; run 'git fetch origin' there first" >&2
    exit 1
  fi

  # Remove a worktree left behind by an interrupted setup.
  if [[ -e "${{SRC_DIR}}" ]]; then
    git -C "${{QUOKKA_ROOT}}" worktree remove --force "${{SRC_DIR}}" 2>/dev/null || rm -rf "${{SRC_DIR}}"
    git -C "${{QUOKKA_ROOT}}" worktree prune
  fi

  git -C "${{QUOKKA_ROOT}}" worktree add --detach "${{SRC_DIR}}" "${{PATCH_BASE_COMMIT}}"
  # Clone submodules from the URLs in the base commit's .gitmodules, not from URLs that
  # the local clone's .git/config may override (that config is shared by all worktrees).
  url_overrides=()
  while read -r key url; do
    name="${{key#submodule.}}"
    url_overrides+=(-c "submodule.${{name%.url}}.url=${{url}}")
  done < <(git -C "${{SRC_DIR}}" config -f .gitmodules --get-regexp '^submodule\\..*\\.url$' || true)
  git -C "${{SRC_DIR}}" ${{url_overrides[@]+"${{url_overrides[@]}}"}} submodule update --init --recursive

  # Apply local code changes (including edits inside submodules) if this reproducer includes any.
  if [[ -s "${{SCRIPT_DIR}}/patches/quokka.patch" ]]; then
    git -C "${{SRC_DIR}}" apply --check "${{SCRIPT_DIR}}/patches/quokka.patch"
    git -C "${{SRC_DIR}}" apply "${{SCRIPT_DIR}}/patches/quokka.patch"
  fi

  # Restore untracked files that cannot be represented by git diff.
  if [[ -d "${{SCRIPT_DIR}}/patches/untracked-files" ]]; then
    cp -R "${{SCRIPT_DIR}}/patches/untracked-files/." "${{SRC_DIR}}/"
  fi

  touch "${{SETUP_MARKER}}"
fi

if [[ ! -d "${{SRC_DIR}}/build/${{PRESET}}" ]]; then
  "${{SRC_DIR}}/scripts/bash/quokka" config --root "${{SRC_DIR}}" -d "${{PRESET}}"
fi
"${{SRC_DIR}}/scripts/bash/quokka" build --root "${{SRC_DIR}}" -d "${{PRESET}}" "${{PROBLEM}}"

cd "${{SCRIPT_DIR}}"
mpirun -np "${{MPI_RANKS}}" "${{SRC_DIR}}/build/${{PRESET}}/src/problems/${{PROBLEM}}/${{PROBLEM}}" "${{INPUT}}"
"""


def create_reproducer(args: argparse.Namespace) -> Path:
    worktree = ensure_quokka_worktree(args.worktree)
    slug = sanitize_slug(args.description)
    output_root = (args.output_dir if args.output_dir is not None else worktree / "tests").resolve()
    repro_dir = output_root / f"{REPRODUCER_PREFIX}{slug}"
    untracked_files_dir = repro_dir / "patches" / "untracked-files"

    if repro_dir.exists():
        if not args.force:
            raise FileExistsError(f"{repro_dir} already exists; pass --force to overwrite it")
        shutil.rmtree(repro_dir)

    branch = git_output(worktree, ["branch", "--show-current"]) or "(detached HEAD)"
    commit = git_output(worktree, ["rev-parse", "HEAD"])
    base_commit = find_base_commit(worktree, args.base_ref)
    status = git_output(worktree, ["status", "--short"])
    untracked = get_untracked_files(worktree, output_root)

    input_dir = repro_dir / "input"
    data_dir = repro_dir / "data"
    patches_dir = repro_dir / "patches"
    output_dir = repro_dir / "output"
    for directory in (input_dir, data_dir, patches_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # --submodule=diff records edits inside submodules as ordinary file diffs; the default
    # format only records "Subproject commit ...-dirty", which loses the edits.
    diff = run_git(worktree, ["diff", "--submodule=diff", "--binary", base_commit]).stdout
    write_text(patches_dir / "quokka.patch", diff)
    write_text(
        patches_dir / "README.md",
        f"quokka.patch was generated with: git diff --submodule=diff --binary {base_commit}\n"
        f"Base ref used to find that commit: {args.base_ref}\n"
        f"Base commit is the merge-base of HEAD and {args.base_ref}: {base_commit}\n"
        "Place additional patch notes here if the diff needs explanation.\n",
    )
    copy_untracked_files(untracked, untracked_files_dir, worktree)

    write_text(repro_dir / "README.md", build_readme(slug, worktree, branch, commit, args.base_ref, base_commit, status, untracked))
    write_text(repro_dir / "run.sh", build_run_script(args.base_ref, base_commit), executable=True)
    write_text(input_dir / "README.md", "Put the complete Quokka input deck here, including every required parameter.\n")
    write_text(data_dir / "README.md", "Put required data tables and auxiliary input files here.\n")
    write_text(output_dir / "failure.log", "Paste the full failing terminal output here.\n")

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
        default=None,
        help="Directory where the reproducer directory is created (default: tests/ in the worktree, which is gitignored)",
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
    tarball = repro_dir.parent / f"{repro_dir.name}.tar.gz"
    print(f"after reviewing the contents: tar -C {repro_dir.parent} --exclude quokka-src -czf {tarball} {repro_dir.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
