from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from quokka.core.constants import AMREX_TIMESTEP_BANNER_RE, NINJA_PROGRESS_RE

from quokka.core.errors import DiagnosticError

from quokka.core.subprocess import shell_join

from quokka.core.types import CommandResult

from quokka.output.json import error_payload, success_payload

def emit_notice(context: CliContext, message: str) -> None:
    if context.json_output:
        return
    print(message, file=sys.stderr, flush=True)

def sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return sanitized or "all"

def command_log_path(context: CliContext, command: str, label: str) -> Path:
    runtime_dir = context.resolve_runtime_dir(command)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    profile = sanitize_label(context.profile_name() or "no-profile")
    return runtime_dir / "runs" / "{}-{}-{}-{}.log".format(timestamp, sanitize_label(command), profile, sanitize_label(label))

def ctest_compact_console_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("Start "):
        return True
    if stripped.startswith("Test project "):
        return True
    if stripped.startswith("The following tests passed:"):
        return True
    if stripped.startswith("The following tests FAILED:"):
        return True
    if stripped.startswith("Total Test time"):
        return True
    if stripped.startswith("Errors while running CTest"):
        return True
    if stripped.startswith("Output from these tests are in:"):
        return True
    if "tests passed" in stripped and "%" in stripped:
        return True
    if re.match(r"^\d+/\d+\s+Test\s+#\d+:", stripped):
        return True
    return False

def make_ctest_stream_console_filter(*, interval_seconds: float = 5.0) -> Any:
    state = {"last_emit": 0.0}

    def reporter(line: str) -> Any:
        stripped = line.strip()
        if not stripped or re.match(r"^\d+:\s*$", stripped):
            return None

        match = AMREX_TIMESTEP_BANNER_RE.match(stripped)
        if match is None:
            return True

        step = int(match.group("step"))
        progress = match.group("progress").strip()
        now = time.monotonic()
        if step == 1 or progress == "100%" or (now - state["last_emit"]) >= interval_seconds:
            state["last_emit"] = now
            return line
        return None

    return reporter

def make_ninja_progress_heartbeat(label: str, *, interval_seconds: float = 5.0) -> Any:
    state = {"last_emit": 0.0}

    def reporter(line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped:
            return None

        match = NINJA_PROGRESS_RE.match(stripped)
        if match is None:
            if stripped.startswith(("FAILED:", "ninja: build stopped")):
                return stripped
            return None

        current = int(match.group(1))
        total = int(match.group(2))
        target = match.group(3).strip()
        now = time.monotonic()
        if current == 1 or current == total or (now - state["last_emit"]) >= interval_seconds:
            state["last_emit"] = now
            return "{} heartbeat: [{}/{}] {}".format(label, current, total, target)
        return None

    return reporter

def format_result(result: CommandResult, as_json: bool) -> str:
    if not as_json:
        return result.text
    payload = {
        "schema": 1,
        "ok": True,
        "command": result.command,
        "profile": result.profile,
        "resource": result.resource,
        "diagnostic": None,
        "data": result.data,
    }
    return json.dumps(payload, indent=2, sort_keys=True)

def bootstrap_hint_command(profile: Optional[str], *, fix: bool = False, include_optional: bool = False) -> str:
    args = ["quokka", "bootstrap"]
    if fix:
        args.append("--fix")
    if include_optional:
        args.append("--include-optional")
    if profile:
        args.extend(["--profile", profile])
    return shell_join(args)

def doctor_hint_command(profile: Optional[str], topic: Optional[str] = None) -> Optional[str]:
    if not profile:
        return None
    args = ["quokka", "doctor"]
    if topic is not None:
        args.append(topic)
    args.extend(["--profile", profile])
    return shell_join(args)

def stream_test_hint_command(profile: Optional[str], resource: Optional[Dict[str, Any]]) -> Optional[str]:
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

def diagnostic_hints(error: DiagnosticError, command: Optional[str], profile: Optional[str]) -> List[str]:
    effective_command = error.command or command
    effective_profile = error.profile or profile
    hints: List[str] = []
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

def error_result(error: DiagnosticError, as_json: bool, command: Optional[str], profile: Optional[str]) -> str:
    hints = diagnostic_hints(error, command, profile)
    if not as_json:
        if not hints:
            return error.args[0]
        return "{}\nHints:\n- {}".format(error.args[0], "\n- ".join(hints))
    payload = {
        "schema": 1,
        "ok": False,
        "command": error.command or command,
        "profile": error.profile or profile,
        "resource": error.resource,
        "diagnostic": {
            "id": error.diagnostic_id,
            "exit_code": error.exit_code,
            "message": error.args[0],
            "details": error.details,
            "hints": hints,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)
