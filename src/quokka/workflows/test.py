from __future__ import annotations

import re
from pathlib import Path

from quokka.core.errors import DiagnosticError
from quokka.core.subprocess import run_command, run_command_compact_logged
from quokka.core.types import CommandResult, TestRequest
from quokka.model.files import resolve_input_argument
from quokka.model.selectors import ctest_selection
from quokka.model.tests import expectation_summary_for_test, problem_for_test
from quokka.output.console import command_log_path, ctest_compact_console_line, emit_notice, make_ctest_stream_console_filter
from quokka.project.context import CliContext
from quokka.project.state import acquire_lock, ensure_no_conflicting_locks, is_build_configured, state_for_artifact
from quokka.tools.ctest import observed_metrics_from_lasttest
from quokka.workflows.build import ensure_profile_configured, perform_build
from quokka.workflows.common import ensure_artifact_ready


def run_test(context: CliContext, request: TestRequest) -> CommandResult:
    profile = context.require_profile("test")
    if (request.stream or request.compact_stream) and request.json_output:
        raise DiagnosticError(
            "USAGE_ERROR",
            "--stream and --compact-stream cannot be combined with --json.",
            command="test",
            profile=context.profile_name(),
        )
    if request.stream and request.compact_stream:
        raise DiagnosticError(
            "USAGE_ERROR",
            "--stream and --compact-stream are mutually exclusive.",
            command="test",
            profile=context.profile_name(),
        )

    context.resolve_runtime_dir("test")
    context.open_db("test")
    ensure_no_conflicting_locks(context, ("build", "run"), "test")
    test_selector_label = request.ctest_regex or request.test_name or "all"
    compact_log = command_log_path(context, "test", test_selector_label) if request.compact_stream else None
    needs_initial_configure = request.build_if_needed and not is_build_configured(profile.build_dir)
    if needs_initial_configure:
        message = "Preparing profile {}: first build will configure CMake and compile dependencies before running tests.".format(
            context.profile_name()
        )
        if compact_log is not None:
            message = "{} Full log: {}".format(message, compact_log)
        emit_notice(context, message)
    if request.build_if_needed:
        ensure_profile_configured(context, "test", compact_log_path=compact_log)
    tests = ctest_selection(context, request)

    unique_targets: list[str] = []
    target_inputs: dict[str, Path | None] = {}
    target_states: dict[str, str] = {}
    for test in tests:
        problem = problem_for_test(test)
        if problem not in unique_targets:
            unique_targets.append(problem)
        target_inputs[problem] = resolve_input_argument(test.command[1:], test.working_directory, context.worktree_root)
    for target in unique_targets:
        state, _ = state_for_artifact(context, target, "test", target_inputs.get(target))
        target_states[target] = state
    needs_target_build = any(state != "ready" for state in target_states.values())

    if request.build_if_needed and not needs_initial_configure:
        repair_targets = [target for target in unique_targets if target_states.get(target) != "ready"]
        if repair_targets:
            message = "Building or refreshing test target(s) before CTest: {}.".format(", ".join(repair_targets))
            if compact_log is not None:
                message = "{} Full log: {}".format(message, compact_log)
            emit_notice(context, message)

    if request.build_if_needed and (needs_initial_configure or needs_target_build):
        perform_build(context, unique_targets, reconfigure=False, compact_log_path=compact_log)

    for target in unique_targets:
        ensure_artifact_ready(context, target, "test", target_inputs.get(target), build_if_needed=False)

    ctest_args = ["ctest", "--test-dir", str(profile.build_dir)]
    if request.stream or request.compact_stream:
        ctest_args.extend(["--progress", "--verbose"])
    else:
        ctest_args.append("--output-on-failure")
    if request.ctest_regex:
        ctest_args.extend(["-R", request.ctest_regex])
        resource_name = request.ctest_regex
        resource_selector = "regex"
    elif request.test_name:
        ctest_args.extend(["-R", "^{}$".format(re.escape(request.test_name))])
        resource_name = request.test_name
        resource_selector = "name"
    else:
        resource_name = "*"
        resource_selector = "all"

    test_resource = {"kind": "test", "name": resource_name, "selector": resource_selector}

    with acquire_lock(context, "run", "test"):
        if request.compact_stream:
            assert compact_log is not None
            emit_notice(context, "Running CTest with compact progress output. Full log: {}".format(compact_log))
            run_command_compact_logged(
                ctest_args,
                command="test",
                profile=context.profile_name(),
                resource=test_resource,
                log_path=compact_log,
                echo_filter=ctest_compact_console_line,
            )
        elif request.stream:
            emit_notice(context, "Running CTest with live output. Repetitive timestep banners are throttled to periodic heartbeats.")
            run_command(
                ctest_args,
                command="test",
                profile=context.profile_name(),
                resource=test_resource,
                capture_output=context.json_output,
                echo_filter=make_ctest_stream_console_filter(),
            )
        else:
            run_command(
                ctest_args,
                command="test",
                profile=context.profile_name(),
                resource=test_resource,
                capture_output=context.json_output,
            )

    expectations = []
    for test in tests:
        expectation = expectation_summary_for_test(context, test)
        if expectation is not None:
            expectations.append(expectation)
    observed_metrics: list[dict[str, object]] = []
    if request.test_name or (request.ctest_regex and len(tests) <= 5):
        observed_metrics = observed_metrics_from_lasttest(profile.build_dir, tests)

    data = {
        "selected_tests": [test.name for test in tests],
        "build_dir": str(profile.build_dir),
        "streaming": request.stream,
        "compact_stream": request.compact_stream,
        "log_path": None if compact_log is None else str(compact_log),
        "expectations": expectations,
        "observed_metrics": observed_metrics,
    }
    text = "Ran {} test(s) in profile {}{}.".format(
        len(tests),
        context.profile_name(),
        " with streaming output" if request.stream else (" with compact streaming output" if request.compact_stream else ""),
    )
    if expectations:
        text += "\nExpectations:"
        for expectation in expectations:
            text += "\n- {}: {}".format(expectation["test"], expectation["summary"])
    if observed_metrics:
        text += "\nObserved metrics:"
        single_test = len(observed_metrics) == 1
        for observed in observed_metrics:
            for line in observed["lines"]:
                if single_test:
                    text += "\n- {}".format(line)
                else:
                    text += "\n- {}: {}".format(observed["test"], line)
    if compact_log is not None:
        text += "\nFull log: {}".format(compact_log)
    return CommandResult("test", context.profile_name(), test_resource, data, text)
