from __future__ import annotations

from quokka.core.errors import DiagnosticError
from quokka.core.result import CommandResult
from quokka.core.types import TidyRequest
from quokka.model.files import select_tidy_files
from quokka.model.selectors import resolve_tidy_selector
from quokka.project.context import CliContext
from quokka.project.discovery import compile_commands_path
from quokka.project.state import ensure_no_conflicting_locks
from quokka.tools.clang_tidy import run_clang_tidy
from quokka.workflows.common import summarize_path_examples


def run_tidy(context: CliContext, request: TidyRequest) -> CommandResult:
    profile = context.require_profile("tidy")
    selector = resolve_tidy_selector(request.selector or "changed", profile=context.profile_name())
    context.resolve_runtime_dir("tidy")
    ensure_no_conflicting_locks(context, ("build",), "tidy")

    compile_commands = compile_commands_path(profile.build_dir)
    if not compile_commands.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' does not have compile_commands.json yet.".format(context.profile_name()),
            command="tidy",
            profile=context.profile_name(),
            details={"compile_commands": str(compile_commands)},
        )

    selection = select_tidy_files(context, selector)
    if not selection.files:
        data = {
            "build_dir": str(profile.build_dir),
            "selector": selector,
            "fix": request.fix,
            "files": [],
            "skipped_files": selection.skipped_files,
            "no_op": True,
        }
        text = "No files selected for clang-tidy."
        if selection.skipped_files:
            text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(selection.skipped_files))
        return CommandResult("tidy", context.profile_name(), None, data, text)

    run_clang_tidy(context, selector, fix=request.fix)
    data = {
        "build_dir": str(profile.build_dir),
        "selector": selector,
        "fix": request.fix,
        "files": selection.files,
        "skipped_files": selection.skipped_files,
    }
    text = "Ran clang-tidy wrapper for profile {} with selector '{}'.".format(context.profile_name(), selector)
    if selection.skipped_files:
        text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(selection.skipped_files))
    return CommandResult("tidy", context.profile_name(), None, data, text)
