from __future__ import annotations

import json
from typing import Any

from quokka.core.errors import DiagnosticError
from quokka.core.subprocess import shell_join
from quokka.output.json import error_payload


def bootstrap_hint_command(profile: str | None, *, fix: bool = False, include_optional: bool = False) -> str:
    args = ["quokka", "bootstrap"]
    if fix:
        args.append("--fix")
    if include_optional:
        args.append("--include-optional")
    if profile:
        args.extend(["--profile", profile])
    return shell_join(args)


def doctor_hint_command(profile: str | None, topic: str | None = None) -> str | None:
    if not profile:
        return None
    args = ["quokka", "doctor"]
    if topic is not None:
        args.append(topic)
    args.extend(["--profile", profile])
    return shell_join(args)


def stream_test_hint_command(profile: str | None, resource: dict[str, Any] | None) -> str | None:
    if not profile:
        return None
    args = ["quokka", "test"]
    selector = None if resource is None else resource.get("selector")
    resource_name = None if resource is None else resource.get("name")
    if selector == "name" and isinstance(resource_name, str) and resource_name != "*":
        args.append(resource_name)
    elif selector == "regex" and isinstance(resource_name, str) and resource_name != "*":
        args.extend(["--ctest-regex", resource_name])
    args.extend(["--profile", profile, "--stream"])
    return shell_join(args)


def diagnostic_hints(error: DiagnosticError, command: str | None, profile: str | None) -> list[str]:
    effective_command = error.command or command
    effective_profile = error.profile or profile
    hints: list[str] = []
    log_path = error.details.get("log_path")

    doctor_command = None
    if error.diagnostic_id == "RESOURCE_LOCKED":
        doctor_command = doctor_hint_command(effective_profile, "locking")
    elif error.diagnostic_id in {"CONFIGURE_DRIFT", "PROFILE_UNCONFIGURED", "MISSING_ARTIFACT", "STALE_ARTIFACT"}:
        doctor_command = doctor_hint_command(effective_profile, "profile")
    elif error.diagnostic_id in {"TOOL_FAILED", "EXECUTOR_UNAVAILABLE", "STATE_CORRUPT"}:
        doctor_command = doctor_hint_command(effective_profile, "all")

    if doctor_command is not None:
        hints.append("Inspect the current environment with: {}".format(doctor_command))

    if effective_command == "test" and error.diagnostic_id == "TOOL_FAILED":
        stream_command = stream_test_hint_command(effective_profile, error.resource)
        if stream_command is not None:
            hints.append("For live CTest output, rerun with: {}".format(stream_command))

    if error.diagnostic_id == "PRE_COMMIT_UNAVAILABLE":
        bootstrap_command = error.details.get("bootstrap_command")
        if not isinstance(bootstrap_command, str) or not bootstrap_command:
            bootstrap_command = bootstrap_hint_command(effective_profile, fix=True)
        hints.append("One-step fix: {}".format(bootstrap_command))
        install_commands = error.details.get("install_commands")
        if isinstance(install_commands, list):
            for command_text in install_commands:
                if isinstance(command_text, str) and command_text:
                    hints.append("Install pre-commit with: {}".format(command_text))
        helper_script = error.details.get("helper_script")
        if isinstance(helper_script, str) and helper_script:
            hints.append("The repository formatter helper can install it interactively: {}".format(helper_script))

    if isinstance(log_path, str) and log_path:
        hints.append("Full command log: {}".format(log_path))

    return hints


def error_result(error: DiagnosticError, as_json: bool, command: str | None, profile: str | None) -> str:
    hints = diagnostic_hints(error, command, profile)
    if not as_json:
        if not hints:
            return error.args[0]
        return "{}\nHints:\n- {}".format(error.args[0], "\n- ".join(hints))
    return json.dumps(error_payload(error, command, profile, hints), indent=2, sort_keys=True)
