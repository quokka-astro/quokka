from __future__ import annotations

from pathlib import Path

from quokka.core.errors import DiagnosticError
from quokka.core.subprocess import run_command, run_command_compact_logged
from quokka.core.types import BuildRequest, CommandResult
from quokka.model.files import resolve_buildtree_binary
from quokka.model.tests import discover_problems
from quokka.output.console import emit_notice, make_ninja_progress_heartbeat
from quokka.project.context import CliContext
from quokka.project.state import acquire_lock, ensure_buildtree_state_layout, is_build_configured, write_artifact_receipt, write_configure_receipt, write_profile_receipt, write_schema_receipt
from quokka.tools.cmake import format_define_mismatch_summary, profile_define_state


def maybe_reconfigure(context: CliContext, command: str, reconfigure: bool, *, compact_log_path: Path | None = None) -> dict[str, object]:
    profile = context.require_profile(command)
    ensure_buildtree_state_layout(profile.build_dir)
    write_schema_receipt(profile)
    write_profile_receipt(context, command)

    needs_configure = reconfigure or not is_build_configured(profile.build_dir)
    if needs_configure:
        args = ["cmake", "-S", str(context.worktree_root), "-B", str(profile.build_dir), "-G", profile.generator]
        for key in sorted(profile.defines):
            args.append("-D{}={}".format(key, profile.defines[key]))
        if compact_log_path is not None:
            emit_notice(context, "Configuring profile {}. Full log: {}".format(context.profile_name(), compact_log_path))
            run_command_compact_logged(
                args,
                command=command,
                profile=context.profile_name(),
                log_path=compact_log_path,
            )
        else:
            run_command(args, command=command, profile=context.profile_name(), capture_output=context.json_output)

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
        return maybe_reconfigure(context, command, reconfigure=False, compact_log_path=compact_log_path)


def perform_build(context: CliContext, targets: tuple[str, ...] | list[str], reconfigure: bool, *, compact_log_path: Path | None = None) -> dict[str, object]:
    command = "build"
    profile = context.require_profile(command)
    context.resolve_runtime_dir(command)
    context.open_db(command)

    with acquire_lock(context, "build", command):
        configure_receipt = maybe_reconfigure(context, command, reconfigure, compact_log_path=compact_log_path)
        build_args = ["cmake", "--build", str(profile.build_dir)]
        if targets:
            build_args.extend(["--target"] + list(targets))
        if compact_log_path is not None:
            target_label = ", ".join(targets) if targets else "default target set"
            emit_notice(context, "Building {} in profile {}. Full log: {}".format(target_label, context.profile_name(), compact_log_path))
            run_command_compact_logged(
                build_args,
                command=command,
                profile=context.profile_name(),
                log_path=compact_log_path,
                echo_filter=make_ninja_progress_heartbeat("Build"),
            )
        else:
            run_command(build_args, command=command, profile=context.profile_name(), capture_output=context.json_output)

        requested = list(dict.fromkeys(targets)) if targets else discover_problems(profile.build_dir, command, context.profile_name())
        receipts_written: list[str] = []
        for artifact_id in requested:
            binary_path = resolve_buildtree_binary(context, artifact_id, command)
            if binary_path is None or not binary_path.exists():
                continue
            write_artifact_receipt(context, artifact_id, binary_path, command)
            receipts_written.append(artifact_id)

    return {
        "build_dir": str(profile.build_dir),
        "configure_fingerprint": configure_receipt["configure_fingerprint"],
        "targets": list(targets),
        "receipts_written": receipts_written,
    }


def run_build(context: CliContext, request: BuildRequest) -> CommandResult:
    data = perform_build(context, request.targets, reconfigure=request.reconfigure)
    text_targets = ", ".join(data["receipts_written"]) if data["receipts_written"] else "(no problem receipts updated)"
    text = "Built profile {} in {}.\nReceipts updated: {}".format(context.profile_name(), data["build_dir"], text_targets)
    return CommandResult("build", context.profile_name(), None, data, text)
