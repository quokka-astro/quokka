from __future__ import annotations

from pathlib import Path

from quokka.core.result import CommandResult
from quokka.core.types import BuildRequest
from quokka.model.files import resolve_buildtree_binary
from quokka.model.targets import discover_problems
from quokka.output.console import emit_notice, make_ninja_progress_heartbeat
from quokka.project.context import CliContext
from quokka.project.state import acquire_lock, write_artifact_receipt
from quokka.tools.cmake import build_targets
from quokka.workflows.configure import perform_configure


def perform_build(context: CliContext, targets: tuple[str, ...] | list[str], reconfigure: bool, *, compact_log_path: Path | None = None) -> dict[str, object]:
    command = "build"
    profile = context.require_profile(command)
    context.resolve_runtime_dir(command)
    context.open_db(command)

    with acquire_lock(context, "build", command):
        configure_receipt = perform_configure(context, command, reconfigure, compact_log_path=compact_log_path)
        if compact_log_path is not None:
            target_label = ", ".join(targets) if targets else "default target set"
            emit_notice(context, "Building {} in profile {}. Full log: {}".format(target_label, context.profile_name(), compact_log_path))
        build_targets(
            profile,
            command,
            context.profile_name(),
            targets=targets,
            capture_output=context.json_output,
            compact_log_path=compact_log_path,
            echo_filter=make_ninja_progress_heartbeat("Build"),
        )

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
