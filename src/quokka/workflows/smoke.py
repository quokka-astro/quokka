from __future__ import annotations

from quokka.core.errors import DiagnosticError
from quokka.core.types import CommandResult, DoctorRequest, SmokeRequest, TestRequest
from quokka.output.console import emit_notice
from quokka.project.context import CliContext
from quokka.workflows.doctor import run_doctor
from quokka.workflows.test import run_test


def run_smoke(context: CliContext, request: SmokeRequest) -> CommandResult:
    if (request.stream or request.compact_stream) and request.json_output:
        raise DiagnosticError(
            "USAGE_ERROR",
            "--stream and --compact-stream cannot be combined with --json.",
            command="smoke",
            profile=context.profile_name(),
        )
    if request.stream and request.compact_stream:
        raise DiagnosticError(
            "USAGE_ERROR",
            "--stream and --compact-stream are mutually exclusive.",
            command="smoke",
            profile=context.profile_name(),
        )

    runtime_result = run_doctor(context, DoctorRequest(profile=request.profile, topic="runtime"))
    profile_result = run_doctor(context, DoctorRequest(profile=request.profile, topic="profile"))
    if not context.json_output:
        emit_notice(context, runtime_result.text)
        emit_notice(context, profile_result.text)

    test_request = TestRequest(
        profile=request.profile,
        json_output=request.json_output,
        test_name=request.test_name or "ODEIntegration",
        ctest_regex=None,
        build_if_needed=True,
        stream=request.stream,
        compact_stream=request.compact_stream,
    )
    test_result = run_test(context, test_request)
    data = {
        "runtime": runtime_result.data,
        "profile": profile_result.data,
        "test": test_result.data,
    }
    text = "Smoke test target: {} in profile {}.\n{}".format(test_request.test_name, context.profile_name(), test_result.text)
    return CommandResult("smoke", context.profile_name(), {"kind": "test", "name": test_request.test_name}, data, text)
