from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from quokka.core.errors import DiagnosticError
from quokka.project.context import CliContext
from quokka.project.state import state_for_artifact
from quokka.tools.ctest import extract_metric_lines


def summarize_path_examples(paths: Sequence[str], *, limit: int = 3) -> str:
    shown = list(paths[:limit])
    hidden = len(paths) - len(shown)
    suffix = " (+{} more)".format(hidden) if hidden > 0 else ""
    return ", ".join(shown) + suffix


def summarize_runtime_output(stdout: str, stderr: str) -> list[str]:
    combined = stdout.splitlines() + stderr.splitlines()
    metrics = extract_metric_lines(combined)
    if metrics:
        return metrics

    fallback: list[str] = []
    seen = set()
    for raw_line in combined:
        line = raw_line.strip()
        if not line or line in seen:
            continue
        lower = line.lower()
        if lower.startswith(
            (
                "initializing amrex",
                "mpi initialized",
                "amrex ",
                "tinyprofiler",
                "unused parmparse",
                "pinned memory",
                "cpu memory",
                "name ",
            )
        ):
            continue
        fallback.append(line)
        seen.add(line)
        if len(fallback) >= 5:
            break
    return fallback


def ensure_artifact_ready(
    context: CliContext,
    artifact_id: str,
    command: str,
    input_path: Path | None,
    build_if_needed: bool,
) -> dict[str, Any]:
    from quokka.workflows.build import perform_build

    state, details = state_for_artifact(context, artifact_id, command, input_path)
    if state == "ready":
        return details

    if build_if_needed:
        perform_build(context, [artifact_id], reconfigure=False)
        state, details = state_for_artifact(context, artifact_id, command, input_path)
        if state == "ready":
            return details

    resource = {"kind": "problem", "name": artifact_id}
    if state == "missing":
        raise DiagnosticError(
            "MISSING_ARTIFACT",
            "{} in profile {} is missing and must be built first.".format(artifact_id, context.profile_name()),
            command=command,
            profile=context.profile_name(),
            resource=resource,
            details=details,
        )
    if state == "stale_configure":
        raise DiagnosticError(
            "CONFIGURE_DRIFT",
            "{} in profile {} no longer matches the active build configuration.".format(artifact_id, context.profile_name()),
            command=command,
            profile=context.profile_name(),
            resource=resource,
            details=details,
        )
    if state == "stale_source":
        raise DiagnosticError(
            "STALE_ARTIFACT",
            "{} in profile {} is stale and must be rebuilt before it can run.".format(artifact_id, context.profile_name()),
            command=command,
            profile=context.profile_name(),
            resource=resource,
            details=details,
        )
    raise DiagnosticError(
        "STATE_CORRUPT",
        "{} has unreadable or inconsistent state.".format(artifact_id),
        command=command,
        profile=context.profile_name(),
        resource=resource,
        details=details,
    )
