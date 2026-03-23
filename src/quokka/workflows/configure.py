from __future__ import annotations

from pathlib import Path

from quokka.core.errors import DiagnosticError
from quokka.core.result import CommandResult
from quokka.core.types import ConfigureRequest
from quokka.output.console import emit_notice
from quokka.project.context import CliContext
from quokka.project.state import acquire_lock, ensure_buildtree_state_layout, is_build_configured, write_configure_receipt, write_profile_receipt, write_schema_receipt
from quokka.tools.cmake import configure_project, format_define_mismatch_summary, profile_define_state


def perform_configure(context: CliContext, command: str, reconfigure: bool, *, compact_log_path: Path | None = None) -> dict[str, object]:
    profile = context.require_profile(command)
    ensure_buildtree_state_layout(profile.build_dir)
    write_schema_receipt(profile)
    write_profile_receipt(context, command)

    needs_configure = reconfigure or not is_build_configured(profile.build_dir)
    if needs_configure:
        if compact_log_path is not None:
            emit_notice(context, "Configuring profile {}. Full log: {}".format(context.profile_name(), compact_log_path))
        configure_project(
            context.worktree_root,
            profile,
            command,
            context.profile_name(),
            capture_output=context.json_output,
            compact_log_path=compact_log_path,
        )

    define_state = profile_define_state(profile, command, context.profile_name())
    if define_state["mismatches"]:
        raise DiagnosticError(
            "CONFIGURE_DRIFT",
            "Profile '{}' requested defines do not match the configured CMake cache: {}.".format(
                context.profile_name(), format_define_mismatch_summary(define_state["mismatches"])
            ),
            command=command,
            profile=context.profile_name(),
            details=define_state,
        )
    return write_configure_receipt(context, command, define_state=define_state)


def ensure_profile_configured(context: CliContext, command: str, *, compact_log_path: Path | None = None) -> dict[str, object]:
    profile = context.require_profile(command)
    if is_build_configured(profile.build_dir):
        return profile_define_state(profile, command, context.profile_name())

    with acquire_lock(context, "build", command):
        return perform_configure(context, command, reconfigure=False, compact_log_path=compact_log_path)


def run_configure(context: CliContext, request: ConfigureRequest) -> CommandResult:
    command = request.command_name
    context.resolve_runtime_dir(command)
    context.open_db(command)
    with acquire_lock(context, "build", command):
        data = perform_configure(context, command, request.reconfigure)
    text = "Configured profile {} in {}.".format(context.profile_name(), context.require_profile(command).build_dir)
    return CommandResult(command, context.profile_name(), None, data, text)
