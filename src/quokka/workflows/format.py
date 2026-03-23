from __future__ import annotations

from quokka.core.constants import CLANG_FORMAT_FILE_EXTENSIONS
from quokka.core.subprocess import command_output
from quokka.core.types import CommandResult, FormatRequest
from quokka.model.selectors import resolve_format_selector
from quokka.project.context import CliContext
from quokka.project.state import ensure_no_conflicting_locks
from quokka.tools.clang_format import clang_format_files
from quokka.vcs.git import git_changed_files
from quokka.workflows.common import summarize_path_examples


def run_format(context: CliContext, request: FormatRequest) -> CommandResult:
    selector = resolve_format_selector(request.selector or "changed")
    context.resolve_runtime_dir("format")
    ensure_no_conflicting_locks(context, ("build",), "format")

    if selector == "all":
        tracked_files = command_output(["git", "ls-files"], cwd=context.worktree_root, command="format", profile=None).splitlines()
        files = clang_format_files(context, tracked_files)
        data = {"selector": selector, "files": files, "all_files": True, "engine": "clang-format"}
        text = "Ran clang-format directly over {} tracked file(s).".format(len(files))
        return CommandResult("format", None, None, data, text)

    changed_files = git_changed_files(context.worktree_root, selector, "format", None)
    files = [path for path in changed_files if path.endswith(CLANG_FORMAT_FILE_EXTENSIONS)]
    skipped_files = [path for path in changed_files if not path.endswith(CLANG_FORMAT_FILE_EXTENSIONS)]
    if not files:
        data = {"selector": selector, "files": [], "skipped_files": skipped_files, "no_op": True}
        text = "No C/C++ files selected for formatting."
        if skipped_files:
            text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(skipped_files))
        return CommandResult("format", None, None, data, text)

    clang_format_files(context, files)
    data = {"selector": selector, "files": files, "skipped_files": skipped_files, "engine": "clang-format"}
    text = "Ran clang-format directly on {} file(s).".format(len(files))
    if skipped_files:
        text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(skipped_files))
    return CommandResult("format", None, None, data, text)
