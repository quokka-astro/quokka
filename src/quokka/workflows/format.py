from __future__ import annotations

from quokka.core.result import CommandResult
from quokka.core.types import FormatRequest
from quokka.model.files import select_format_files
from quokka.model.selectors import resolve_format_selector
from quokka.project.context import CliContext
from quokka.project.state import ensure_no_conflicting_locks
from quokka.tools.clang_format import clang_format_files
from quokka.workflows.common import summarize_path_examples


def run_format(context: CliContext, request: FormatRequest) -> CommandResult:
    selector = resolve_format_selector(request.selector or "changed")
    context.resolve_runtime_dir("format")
    ensure_no_conflicting_locks(context, ("build",), "format")

    selection = select_format_files(context, selector)
    if not selection.files:
        data = {"selector": selector, "files": [], "skipped_files": selection.skipped_files, "no_op": True}
        text = "No C/C++ files selected for formatting."
        if selection.skipped_files:
            text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(selection.skipped_files))
        return CommandResult("format", None, None, data, text)

    clang_format_files(context, selection.files)
    data = {
        "selector": selector,
        "files": selection.files,
        "skipped_files": selection.skipped_files,
        "all_files": selection.all_files,
        "engine": "clang-format",
    }
    if selection.all_files:
        text = "Ran clang-format directly over {} tracked file(s).".format(len(selection.files))
    else:
        text = "Ran clang-format directly on {} file(s).".format(len(selection.files))
    if selection.skipped_files:
        text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(selection.skipped_files))
    return CommandResult("format", None, None, data, text)
