from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Sequence

from quokka.core.constants import CLANG_FORMAT_FILE_EXTENSIONS, CLANG_TIDY_FILE_EXTENSIONS
from quokka.core.errors import DiagnosticError
from quokka.project.context import CliContext
from quokka.project.root import is_subpath
from quokka.vcs.git import git_changed_files, git_tracked_files


@dataclasses.dataclass(frozen=True)
class FileSelection:
    selector: str
    files: list[str]
    skipped_files: list[str]
    all_files: bool = False


def split_files_by_extension(paths: Sequence[str], extensions: tuple[str, ...]) -> tuple[list[str], list[str]]:
    files: list[str] = []
    skipped_files: list[str] = []
    for path in paths:
        if path.endswith(extensions):
            files.append(path)
        else:
            skipped_files.append(path)
    return files, skipped_files


def select_format_files(context: CliContext, selector: str) -> FileSelection:
    if selector == "all":
        tracked_files = git_tracked_files(context.worktree_root, "format")
        files, skipped_files = split_files_by_extension(tracked_files, CLANG_FORMAT_FILE_EXTENSIONS)
        return FileSelection(selector=selector, files=files, skipped_files=skipped_files, all_files=True)

    changed_files = git_changed_files(context.worktree_root, selector, "format", None)
    files, skipped_files = split_files_by_extension(changed_files, CLANG_FORMAT_FILE_EXTENSIONS)
    return FileSelection(selector=selector, files=files, skipped_files=skipped_files)


def select_tidy_files(context: CliContext, selector: str) -> FileSelection:
    changed_files = git_changed_files(context.worktree_root, selector, "tidy", context.profile_name())
    files, skipped_files = split_files_by_extension(changed_files, CLANG_TIDY_FILE_EXTENSIONS)
    return FileSelection(selector=selector, files=files, skipped_files=skipped_files)


def resolve_buildtree_binary(context: CliContext, problem: str, command: str) -> Optional[Path]:
    from quokka.model.tests import discover_tests
    from quokka.project.state import artifact_receipt_path, read_json

    profile = context.require_profile(command)
    receipt_path = artifact_receipt_path(profile.build_dir, problem)
    if receipt_path.exists():
        receipt = read_json(receipt_path, command, context.profile_name())
        binary_path = Path(str(receipt.get("binary_path", "")))
        if binary_path.exists():
            return binary_path

    candidate = profile.build_dir / "src" / "problems" / problem / problem
    if candidate.exists():
        return candidate

    matches = list((profile.build_dir / "src" / "problems").glob("*/{}".format(problem)))
    if matches:
        return matches[0]

    tests = discover_tests(profile.build_dir, command, context.profile_name())
    for test in tests:
        if test.command and Path(test.command[0]).name == problem:
            return Path(test.command[0]).resolve()
    return None


def resolve_input_argument(arguments: Sequence[str], working_directory: Optional[Path], worktree_root: Path) -> Optional[Path]:
    if working_directory is None:
        bases = [worktree_root]
    else:
        bases = [working_directory, worktree_root]
    for arg in arguments:
        candidate = Path(arg)
        for base in bases:
            resolved = candidate if candidate.is_absolute() else (base / candidate).resolve()
            if resolved.exists() and resolved.is_file():
                return resolved
    return None


def default_input_for_problem(context: CliContext, problem: str, command: str) -> Optional[Path]:
    from quokka.model.tests import discover_tests

    profile = context.require_profile(command)
    tests = discover_tests(profile.build_dir, command, context.profile_name())
    for test in tests:
        if test.name == problem and test.command:
            resolved = resolve_input_argument(test.command[1:], test.working_directory, context.worktree_root)
            if resolved is not None:
                return resolved

    candidate = context.worktree_root / "inputs" / "{}.toml".format(problem)
    if candidate.exists():
        return candidate.resolve()
    return None


def resolve_run_input(context: CliContext, problem: str, input_arg: Optional[str], command: str) -> Path:
    if input_arg:
        candidate = Path(input_arg).expanduser()
        if not candidate.is_absolute():
            candidate = (context.worktree_root / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise DiagnosticError(
            "INPUT_REQUIRED",
            "Input file '{}' does not exist.".format(input_arg),
            command=command,
            profile=context.profile_name(),
            resource={"kind": "problem", "name": problem},
            details={"input": input_arg},
        )

    resolved = default_input_for_problem(context, problem, command)
    if resolved is not None:
        return resolved

    raise DiagnosticError(
        "INPUT_REQUIRED",
        "Unable to resolve an input file for '{}'.".format(problem),
        command=command,
        profile=context.profile_name(),
        resource={"kind": "problem", "name": problem},
    )


def relative_or_absolute(path: Path, worktree_root: Path) -> str:
    if is_subpath(path, worktree_root):
        return str(path.relative_to(worktree_root))
    return str(path)
