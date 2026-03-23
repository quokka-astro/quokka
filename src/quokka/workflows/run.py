from __future__ import annotations

from pathlib import Path

from quokka.core.result import CommandResult
from quokka.core.subprocess import run_command, run_command_capture_output
from quokka.core.types import RunRequest
from quokka.model.files import relative_or_absolute, resolve_run_input
from quokka.project.context import CliContext
from quokka.project.state import acquire_lock, ensure_no_conflicting_locks
from quokka.workflows.common import ensure_artifact_ready, summarize_runtime_output
from quokka.workflows.configure import ensure_profile_configured


def run_problem(context: CliContext, request: RunRequest) -> CommandResult:
    resource = {"kind": "problem", "name": request.problem}
    context.resolve_runtime_dir("run")
    context.open_db("run")
    ensure_no_conflicting_locks(context, ("build", "run"), "run")
    if request.build_if_needed:
        ensure_profile_configured(context, "run")
    input_path = resolve_run_input(context, request.problem, request.input, "run")
    readiness = ensure_artifact_ready(context, request.problem, "run", input_path, request.build_if_needed)
    binary_path = Path(readiness["binary_path"])

    with acquire_lock(context, "run", "run"):
        working_dir_value = ((readiness.get("receipt") or {}).get("inputs") or {}).get("default_working_dir")
        if working_dir_value:
            working_dir = (context.worktree_root / str(working_dir_value)).resolve()
        else:
            working_dir = context.worktree_root / "tests"
        command_args = [str(binary_path), str(input_path)]
        runtime_output = {"stdout": "", "stderr": ""}
        if request.verbose_runtime:
            run_command(
                command_args,
                cwd=working_dir,
                command="run",
                profile=context.profile_name(),
                resource=resource,
                capture_output=context.json_output,
            )
        else:
            runtime_output = run_command_capture_output(
                command_args,
                cwd=working_dir,
                command="run",
                profile=context.profile_name(),
                resource=resource,
            )

    summary_lines = summarize_runtime_output(runtime_output["stdout"], runtime_output["stderr"]) if not request.verbose_runtime else []
    data = {
        "binary_path": str(binary_path),
        "input": relative_or_absolute(input_path, context.worktree_root),
        "working_dir": str(working_dir),
        "verbose_runtime": request.verbose_runtime,
        "summary_lines": summary_lines,
    }
    text = "Ran {} with {}.".format(request.problem, data["input"])
    if summary_lines:
        text += "\nObserved metrics:"
        for line in summary_lines:
            text += "\n- {}".format(line)
    elif not request.verbose_runtime:
        text += "\nNo summary metrics were extracted. Re-run with --verbose-runtime for full program output."
    return CommandResult("run", context.profile_name(), resource, data, text)
