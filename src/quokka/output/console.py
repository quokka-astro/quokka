from __future__ import annotations

import datetime as dt
import re
import sys
import time
from pathlib import Path
from typing import Any

from quokka.core.constants import AMREX_TIMESTEP_BANNER_RE, NINJA_PROGRESS_RE


def emit_notice(context: "CliContext", message: str) -> None:
    if context.json_output:
        return
    print(message, file=sys.stderr, flush=True)


def sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return sanitized or "all"


def command_log_path(context: "CliContext", command: str, label: str) -> Path:
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
    return re.match(r"^\d+/\d+\s+Test\s+#\d+:", stripped) is not None


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

    def reporter(line: str) -> str | None:
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
