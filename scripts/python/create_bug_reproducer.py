#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_git(worktree: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=check,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_bytes(worktree: Path, args: list[str]) -> bytes:
    """Run git and return raw stdout, so CRLF line endings and non-UTF-8 content survive."""
    return subprocess.run(["git", "-C", str(worktree), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def split_paths(output: bytes) -> list[str]:
    """Split NUL-separated git path output, keeping undecodable bytes via os.fsdecode."""
    return [os.fsdecode(name) for name in output.split(b"\0") if name]


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
    path.write_text(content, encoding="utf-8", errors="surrogateescape")
    if executable:
        path.chmod(0o755)


def copy_files(files: list[Path], destination: Path) -> None:
    for source in files:
        # absolute() keeps the given name for symlinks; run.sh refers to the copy by that name.
        source_path = source.absolute()
        if not source_path.exists():
            raise FileNotFoundError(f"{source_path} does not exist")
        target = destination / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target)


def copy_untracked_files(paths: list[str], destination: Path, worktree: Path) -> None:
    if not paths:
        return
    for path in paths:
        source = worktree / path
        if not source.is_file():
            print(f"warning: skipping untracked path that is not a regular file: {path}", file=sys.stderr)
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


REPRODUCER_PREFIX = "quokka-reproducer-"


def list_untracked(directory: Path) -> list[str]:
    """Return untracked, non-ignored files in one repository, NUL-separated to preserve unusual names."""
    return split_paths(git_bytes(directory, ["ls-files", "-z", "--others", "--exclude-standard"]))


def get_untracked_files(worktree: Path) -> list[str]:
    """Return untracked files in the worktree and all submodules, skipping earlier reproducers.

    Paths are relative to the worktree. `git ls-files --others` does not recurse into
    submodules, so each submodule is listed separately.
    """
    files = list_untracked(worktree)
    submodules = split_paths(git_bytes(worktree, ["submodule", "foreach", "--recursive", "--quiet", 'printf "%s\\0" "$displaypath"']))
    for submodule in submodules:
        files.extend(f"{submodule}/{name}" for name in list_untracked(worktree / submodule))
    # lstrip(".") also matches the hidden staging directories used while a reproducer is built.
    return [path for path in files if not any(part.lstrip(".").startswith(REPRODUCER_PREFIX) for part in Path(path).parts)]


LARGE_UNTRACKED_BYTES = 10 * 1024 * 1024


def report_untracked(paths: list[str], worktree: Path) -> None:
    """Print the untracked files that will be copied, so large or private files are noticed."""
    if not paths:
        return
    sizes = [(worktree / path).stat().st_size if (worktree / path).is_file() else 0 for path in paths]
    print(f"copying {len(paths)} untracked file(s) into patches/untracked-files/ (pass --no-untracked to skip):", file=sys.stderr)
    for path, size in zip(paths, sizes):
        print(f"  {size / 1024:10.1f} KiB  {path}", file=sys.stderr)
    total = sum(sizes)
    if total > LARGE_UNTRACKED_BYTES:
        print(f"warning: untracked files total {total / 1024**2:.1f} MiB; check them before uploading", file=sys.stderr)


def find_base_commit(worktree: Path, base_ref: str) -> str:
    try:
        run_git(worktree, ["rev-parse", "--verify", base_ref])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"base ref {base_ref!r} was not found; fetch the official Quokka remote or pass --base-ref"
        ) from exc
    try:
        return git_output(worktree, ["merge-base", "HEAD", base_ref])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"HEAD and {base_ref!r} have no common ancestor in this clone; if it is shallow, run 'git fetch --unshallow'"
        ) from exc


UPSTREAM_REPO = "quokka-astro/quokka"


def check_base_is_upstream(worktree: Path, base_commit: str) -> None:
    """Warn when the base commit is not on the official development branch.

    The maintainer's clone must contain the base commit; a commit that exists only in a fork cannot be fetched.
    """
    remotes = git_output(worktree, ["remote"]).split()
    upstream = [r for r in remotes if UPSTREAM_REPO in run_git(worktree, ["remote", "get-url", r], check=False).stdout]
    for remote in upstream:
        ref = f"{remote}/development"
        if run_git(worktree, ["rev-parse", "--verify", "--quiet", ref], check=False).returncode != 0:
            continue
        if run_git(worktree, ["merge-base", "--is-ancestor", base_commit, ref], check=False).returncode == 0:
            return
        print(
            f"warning: base commit {base_commit} is not on {ref}; if you work from a fork, "
            f"run 'git fetch {remote}' and pass --base-ref {ref}",
            file=sys.stderr,
        )
        return
    print(
        f"warning: no remote for {UPSTREAM_REPO} with a development branch was found, so the base commit "
        f"{base_commit} could not be checked; it must exist upstream for maintainers to use the reproducer",
        file=sys.stderr,
    )


def build_readme(
    slug: str,
    branch: str,
    commit: str,
    base_ref: str,
    base_commit: str,
    status: str,
    untracked: list[str],
    no_untracked: bool,
) -> str:
    if no_untracked:
        untracked_text = "- Skipped (`--no-untracked`); untracked files were not checked or copied."
    elif untracked:
        untracked_text = "\n".join(f"- `{path}`" for path in untracked)
    else:
        untracked_text = "- None detected."
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
The executable runs from `quokka-src/tests/`, like `quokka run`, so the `../extern/...` and
`../inputs/...` paths in standard input decks work; `quokka-src/tests/data` links to `data/`,
so input decks can also refer to `data/...`.

## Worktree status at creation time

```text
{status_text}
```

## Untracked files copied separately

{untracked_text}
"""


def build_run_script(base_ref: str, base_commit: str, input_name: str, problem: str) -> str:
    # Values from the command line are shell-quoted so that unusual names cannot break run.sh.
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"

# The patch in patches/quokka.patch was generated relative to this commit.
PATCH_BASE_REF={shlex.quote(base_ref)}
PATCH_BASE_COMMIT={shlex.quote(base_commit)}

# Edit these values before uploading the reproducer.
DEFAULT_PROBLEM={shlex.quote(problem)}
DEFAULT_INPUT={shlex.quote(input_name)}
PRESET="${{PRESET:-3d}}"
PROBLEM="${{PROBLEM:-${{DEFAULT_PROBLEM}}}}"
INPUT="${{INPUT:-${{SCRIPT_DIR}}/input/${{DEFAULT_INPUT}}}}"
MPI_RANKS="${{MPI_RANKS:-1}}"
# Optional environment file for quokka --source, e.g. QUOKKA_SOURCE=default for ~/.config/quokka/quokka.rc.
QUOKKA_SOURCE="${{QUOKKA_SOURCE:-}}"

# The executable runs from another directory, so make a relative INPUT override absolute.
if [[ "${{INPUT}}" != /* ]]; then
  INPUT="${{PWD}}/${{INPUT}}"
fi

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
BUILD_DIR="${{SRC_DIR}}/build/${{PRESET}}"

echo "Patch base ref: ${{PATCH_BASE_REF}}"
echo "Patch base commit: ${{PATCH_BASE_COMMIT}}"
echo "Quokka clone: ${{QUOKKA_ROOT}}"
echo "Reproducer worktree: ${{SRC_DIR}}"

if [[ ! -f "${{SETUP_MARKER}}" ]]; then
  if ! git -C "${{QUOKKA_ROOT}}" cat-file -e "${{PATCH_BASE_COMMIT}}^{{commit}}" 2>/dev/null; then
    echo "error: commit ${{PATCH_BASE_COMMIT}} is not in ${{QUOKKA_ROOT}}; run 'git fetch origin' there first" >&2
    exit 1
  fi

  # Remove a worktree left behind by an interrupted setup, and drop the registration of
  # a quokka-src/ directory that was deleted with rm -rf.
  if [[ -e "${{SRC_DIR}}" ]]; then
    git -C "${{QUOKKA_ROOT}}" worktree remove --force "${{SRC_DIR}}" 2>/dev/null || rm -rf "${{SRC_DIR}}"
  fi
  git -C "${{QUOKKA_ROOT}}" worktree prune

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
    # --whitespace=nowarn keeps a local apply.whitespace=error setting from rejecting the patch.
    git -C "${{SRC_DIR}}" apply --whitespace=nowarn --check "${{SCRIPT_DIR}}/patches/quokka.patch"
    git -C "${{SRC_DIR}}" apply --whitespace=nowarn "${{SCRIPT_DIR}}/patches/quokka.patch"
  fi

  # Restore untracked files that cannot be represented by git diff.
  if [[ -d "${{SCRIPT_DIR}}/patches/untracked-files" ]]; then
    cp -R "${{SCRIPT_DIR}}/patches/untracked-files/." "${{SRC_DIR}}/"
  fi

  # Input decks may refer to the reproducer's data/ directory as data/...
  mkdir -p "${{SRC_DIR}}/tests"
  ln -sfn "${{SCRIPT_DIR}}/data" "${{SRC_DIR}}/tests/data"

  touch "${{SETUP_MARKER}}"
fi

source_args=()
if [[ -n "${{QUOKKA_SOURCE}}" ]]; then
  source_args=(--source "${{QUOKKA_SOURCE}}")
fi

# Configure unless a previous configure finished; quokka config creates the build
# directory before running cmake, so a failed configure leaves it without build.ninja.
if [[ ! -f "${{BUILD_DIR}}/build.ninja" ]]; then
  config_args=()
  if [[ -d "${{BUILD_DIR}}" ]]; then
    config_args=(--delete)
  fi
  "${{SRC_DIR}}/scripts/bash/quokka" config --root "${{SRC_DIR}}" -d "${{PRESET}}" ${{source_args[@]+"${{source_args[@]}}"}} ${{config_args[@]+"${{config_args[@]}}"}}
fi
"${{SRC_DIR}}/scripts/bash/quokka" build --root "${{SRC_DIR}}" -d "${{PRESET}}" ${{source_args[@]+"${{source_args[@]}}"}} "${{PROBLEM}}"

# Run from tests/, like quokka run, so the relative paths in standard input decks resolve.
cd "${{SRC_DIR}}/tests"
mpirun -np "${{MPI_RANKS}}" "${{BUILD_DIR}}/src/problems/${{PROBLEM}}/${{PROBLEM}}" "${{INPUT}}"
"""


# Diff options that keep the patch in the format git apply expects, whatever the user's git config.
DIFF_ARGS = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", "--submodule=diff", "--ignore-submodules=none", "--binary"]


def populate_reproducer(repro_dir: Path, args: argparse.Namespace, worktree: Path, slug: str) -> None:
    branch = git_output(worktree, ["branch", "--show-current"]) or "(detached HEAD)"
    commit = git_output(worktree, ["rev-parse", "HEAD"])
    base_commit = find_base_commit(worktree, args.base_ref)
    check_base_is_upstream(worktree, base_commit)
    status = git_output(worktree, ["-c", "core.quotePath=false", "status", "--short", "--ignore-submodules=none"])
    untracked = [] if args.no_untracked else get_untracked_files(worktree)
    report_untracked(untracked, worktree)

    input_dir = repro_dir / "input"
    data_dir = repro_dir / "data"
    patches_dir = repro_dir / "patches"
    output_dir = repro_dir / "output"
    for directory in (input_dir, data_dir, patches_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # --submodule=diff records edits inside submodules as ordinary file diffs; the default
    # format only records "Subproject commit ...-dirty", which loses the edits.
    patch = git_bytes(worktree, [*DIFF_ARGS, base_commit])
    # Without the old submodule commit, git prints only this header instead of the submodule's changes.
    missing = [line for line in patch.splitlines() if line.startswith(b"Submodule ") and b"(commits not present)" in line]
    if missing:
        names = ", ".join(os.fsdecode(line.split()[1]) for line in missing)
        raise RuntimeError(
            f"cannot diff submodule(s) {names} because the base commit's version is missing from the local clone; "
            "fetch the full history there (e.g. 'git -C <submodule> fetch --unshallow') and retry"
        )
    (patches_dir / "quokka.patch").write_bytes(patch)
    write_text(
        patches_dir / "README.md",
        f"quokka.patch was generated with: git {' '.join(DIFF_ARGS)} {base_commit}\n"
        f"Base ref used to find that commit: {args.base_ref}\n"
        f"Base commit is the merge-base of HEAD and {args.base_ref}: {base_commit}\n"
        "Place additional patch notes here if the diff needs explanation.\n",
    )
    copy_untracked_files(untracked, patches_dir / "untracked-files", worktree)

    write_text(
        repro_dir / "README.md", build_readme(slug, branch, commit, args.base_ref, base_commit, status, untracked, args.no_untracked)
    )
    # Default run.sh to the input deck when exactly one input file is given, and to the problem
    # with the same name when one exists (many decks are not named after their problem).
    input_name, problem = "MyProblem.toml", "MyProblem"
    if len(args.input) == 1 and args.input[0].is_file():
        input_name = args.input[0].absolute().name
        stem = args.input[0].absolute().stem
        if (worktree / "src" / "problems" / stem).is_dir():
            problem = stem
        else:
            print(f"warning: no problem named {stem!r} in src/problems; set PROBLEM in run.sh", file=sys.stderr)
    write_text(repro_dir / "run.sh", build_run_script(args.base_ref, base_commit, input_name, problem), executable=True)
    write_text(output_dir / "failure.log", "Paste the full failing terminal output here.\n")

    copy_files(args.input, input_dir)
    copy_files(args.data, data_dir)
    # Write placeholders last, so they never overwrite a user file with the same name.
    for directory, text in (
        (input_dir, "Put the complete Quokka input deck here, including every required parameter.\n"),
        (data_dir, "Put required data tables and auxiliary input files here.\n"),
    ):
        if not (directory / "README.md").exists():
            write_text(directory / "README.md", text)


def create_reproducer(args: argparse.Namespace) -> Path:
    worktree = ensure_quokka_worktree(args.worktree)
    slug = sanitize_slug(args.description)
    output_root = (args.output_dir if args.output_dir is not None else worktree / "tests").resolve()
    repro_dir = output_root / f"{REPRODUCER_PREFIX}{slug}"

    if repro_dir.exists() and not args.force:
        raise FileExistsError(f"{repro_dir} already exists; pass --force to overwrite it")

    # Build in a staging directory and swap it in at the end, so a failure with --force
    # leaves the existing reproducer (and the reporter's edits to it) in place.
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{repro_dir.name}.", dir=output_root))
    try:
        staging.chmod(0o755)
        populate_reproducer(staging, args, worktree, slug)
        if repro_dir.exists():
            shutil.rmtree(repro_dir)
            # Drop the registration of a quokka-src/ worktree that run.sh may have created.
            run_git(worktree, ["worktree", "prune"])
        staging.rename(repro_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
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
        help="Input file or directory (relative to the current directory) to copy into input/; may be passed multiple times",
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        default=[],
        help="Data file or directory (relative to the current directory) to copy into data/; may be passed multiple times",
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
    parser.add_argument(
        "--no-untracked",
        action="store_true",
        help="Do not copy untracked files from the worktree into patches/untracked-files/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        repro_dir = create_reproducer(args)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        detail = stderr.strip() or f"exit status {exc.returncode}"
        print(f"error: {shlex.join(exc.cmd)} failed: {detail}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"created {repro_dir}")
    print("edit README.md, run.sh, input/, data/, and output/ before uploading")
    tarball = repro_dir.parent / f"{repro_dir.name}.tar.gz"
    # Leave out the worktree that run.sh creates; the run writes its outputs inside it.
    excludes = f"--exclude {shlex.quote(f'{repro_dir.name}/quokka-src')}"
    tar_cmd = f"tar -C {shlex.quote(str(repro_dir.parent))} {excludes} -czf {shlex.quote(str(tarball))} {shlex.quote(repro_dir.name + '/')}"
    print(f"after reviewing the contents: {tar_cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
