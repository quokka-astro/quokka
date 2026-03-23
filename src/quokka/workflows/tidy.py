from __future__ import annotations

from quokka.core.constants import CLANG_TIDY_FILE_EXTENSIONS
from quokka.core.errors import DiagnosticError
from quokka.core.types import CommandResult, TidyRequest
from quokka.model.selectors import resolve_tidy_selector
from quokka.project.context import CliContext
from quokka.project.state import ensure_no_conflicting_locks
from quokka.tools.clang_tidy import run_clang_tidy
from quokka.vcs.git import git_changed_files
from quokka.workflows.common import summarize_path_examples


def run_tidy(context: CliContext, request: TidyRequest) -> CommandResult:
    profile = context.require_profile("tidy")
    selector = resolve_tidy_selector(request.selector or "changed", profile=context.profile_name())
    context.resolve_runtime_dir("tidy")
    ensure_no_conflicting_locks(context, ("build",), "tidy")

    compile_commands = profile.build_dir / "compile_commands.json"
    if not compile_commands.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' does not have compile_commands.json yet.".format(context.profile_name()),
            command="tidy",
            profile=context.profile_name(),
            details={"compile_commands": str(compile_commands)},
        )

    changed_files = git_changed_files(context.worktree_root, selector, "tidy", context.profile_name())
    files = [path for path in changed_files if path.endswith(CLANG_TIDY_FILE_EXTENSIONS)]
    skipped_files = [path for path in changed_files if not path.endswith(CLANG_TIDY_FILE_EXTENSIONS)]
    if not files:
        data = {
            "build_dir": str(profile.build_dir),
            "selector": selector,
            "fix": request.fix,
            "files": [],
            "skipped_files": skipped_files,
            "no_op": True,
        }
        text = "No files selected for clang-tidy."
        if skipped_files:
            text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(skipped_files))
        return CommandResult("tidy", context.profile_name(), None, data, text)

    run_clang_tidy(context, selector, fix=request.fix)
    data = {
        "build_dir": str(profile.build_dir),
        "selector": selector,
        "fix": request.fix,
        "files": files,
        "skipped_files": skipped_files,
    }
    text = "Ran clang-tidy wrapper for profile {} with selector '{}'.".format(context.profile_name(), selector)
    if skipped_files:
        text += "\nSkipped non-C/C++ file(s): {}".format(summarize_path_examples(skipped_files))
    return CommandResult("tidy", context.profile_name(), None, data, text)
