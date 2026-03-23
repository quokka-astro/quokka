from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

from quokka.core.constants import SUBMODULE_PATHS
from quokka.core.errors import DiagnosticError
from quokka.core.subprocess import command_output
from quokka.project.root import is_subpath

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def git_status_lines(worktree_root: Path, paths: Sequence[str]) -> list[str]:
    args = ["git", "status", "--porcelain=v1", "--untracked-files=all", "--"] + list(paths)
    output = command_output(args, cwd=worktree_root, command="status", profile=None)
    if not output:
        return []
    return [line for line in output.splitlines() if line]

def git_rev_parse(worktree_root: Path, rev: str) -> str:
    return command_output(["git", "rev-parse", rev], cwd=worktree_root, command="status", profile=None)

def git_tracked_files(worktree_root: Path, command: str, profile: Optional[str] = None) -> list[str]:
    output = command_output(["git", "ls-files"], cwd=worktree_root, command=command, profile=profile)
    return [line for line in output.splitlines() if line]


def compute_source_fingerprint(worktree_root: Path, input_path: Optional[Path], command: str, profile: Optional[str]) -> str:
    digest = hashlib.sha256()
    head = command_output(["git", "rev-parse", "HEAD"], cwd=worktree_root, command=command, profile=profile)
    digest.update(("HEAD\n" + head + "\n").encode("utf-8"))

    for submodule in SUBMODULE_PATHS:
        submodule_path = worktree_root / submodule
        if (submodule_path / ".git").exists() or submodule_path.exists():
            with contextlib.suppress(DiagnosticError):
                sha = command_output(["git", "-C", str(submodule_path), "rev-parse", "HEAD"], command=command, profile=profile)
                digest.update(("SUBMODULE {} {}\n".format(submodule, sha)).encode("utf-8"))

    pathspecs = ["CMakeLists.txt", "cmake", "src"]
    if input_path is not None and is_subpath(input_path, worktree_root):
        relative_input = str(input_path.relative_to(worktree_root))
        if relative_input not in pathspecs:
            pathspecs.append(relative_input)

    status_lines = git_status_lines(worktree_root, pathspecs)
    for line in status_lines:
        digest.update((line + "\n").encode("utf-8"))
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        candidate = (worktree_root / path_text).resolve()
        if candidate.exists() and candidate.is_file():
            digest.update((path_text + "\0" + file_hash(candidate)).encode("utf-8"))

    if input_path is not None and not is_subpath(input_path, worktree_root):
        digest.update(("EXTERNAL_INPUT {}\n".format(str(input_path))).encode("utf-8"))
        if input_path.exists() and input_path.is_file():
            digest.update(file_hash(input_path).encode("utf-8"))

    return "sha256:" + digest.hexdigest()

def git_metadata(worktree_root: Path, command: str, profile: Optional[str]) -> dict[str, Any]:
    head = command_output(["git", "rev-parse", "HEAD"], cwd=worktree_root, command=command, profile=profile)
    dirty = bool(command_output(["git", "status", "--porcelain", "-uno"], cwd=worktree_root, command=command, profile=profile))
    submodules: dict[str, str] = {}
    for submodule in SUBMODULE_PATHS:
        sub_path = worktree_root / submodule
        with contextlib.suppress(DiagnosticError):
            if sub_path.exists():
                submodules[submodule] = command_output(["git", "-C", str(sub_path), "rev-parse", "HEAD"], command=command, profile=profile)
    return {"head": head, "dirty": dirty, "submodules": submodules}

def git_changed_files(worktree_root: Path, selector: str, command: str, profile: Optional[str]) -> list[str]:
    if selector == "changed":
        output = command_output(["git", "diff", "--name-only", "HEAD"], cwd=worktree_root, command=command, profile=profile)
    elif selector == "previous":
        output = command_output(["git", "diff", "--name-only", "HEAD^"], cwd=worktree_root, command=command, profile=profile)
    elif selector == "origin":
        branch = command_output(["git", "branch", "--show-current"], cwd=worktree_root, command=command, profile=profile)
        output = command_output(
            ["git", "diff", "--name-only", "origin/{}".format(branch)],
            cwd=worktree_root,
            command=command,
            profile=profile,
        )
    elif selector == "dev":
        output = command_output(["git", "diff", "--name-only", "development"], cwd=worktree_root, command=command, profile=profile)
    else:
        raise ValueError(selector)
    return [line for line in output.splitlines() if line]
