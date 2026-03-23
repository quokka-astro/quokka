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

from quokka.core.constants import MAX_DIAGNOSTIC_OUTPUT_CHARS

from quokka.core.errors import DiagnosticError

from quokka.project.root import utc_now

def truncate_output_tail(output: Optional[str]) -> str:
    if not output:
        return ""
    return output[-MAX_DIAGNOSTIC_OUTPUT_CHARS:]

def render_console_lines(rendered: Any, line: str) -> List[str]:
    if isinstance(rendered, str):
        return [rendered] if rendered else []
    if isinstance(rendered, (list, tuple)):
        lines: List[str] = []
        for item in rendered:
            if isinstance(item, str) and item:
                lines.append(item)
            elif item:
                lines.append(line)
        return lines
    if rendered:
        return [line]
    return []

def run_command_compact_logged(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    command: str,
    profile: Optional[str],
    resource: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Path,
    echo_filter: Optional[Any] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "$ {}".format(shell_join(args)),
        "# cwd: {}".format(str(cwd) if cwd is not None else os.getcwd()),
        "# started_at: {}".format(utc_now()),
        "",
    ]
    tail = ""
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(header_lines))
            proc = subprocess.Popen(
                list(args),
                cwd=None if cwd is None else str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                handle.write(raw_line)
                tail = truncate_output_tail(tail + raw_line)
                line = raw_line.rstrip("\n")
                if echo_filter is not None:
                    for console_line in render_console_lines(echo_filter(line), line):
                        print(console_line, file=sys.stderr, flush=True)
            returncode = proc.wait()
            handle.write("\n# finished_at: {}\n".format(utc_now()))
        if returncode != 0:
            raise DiagnosticError(
                "TOOL_FAILED",
                "Command failed: {}".format(shell_join(args)),
                command=command,
                profile=profile,
                resource=resource,
                details={
                    "tool": args[0],
                    "exit_code": returncode,
                    "stdout": tail,
                    "stderr": "",
                    "log_path": str(log_path),
                },
            )
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Required tool '{}' is not available.".format(args[0]),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0]},
        ) from exc

def run_command_capture_output(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    command: str,
    profile: Optional[str],
    resource: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    try:
        proc = subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            check=False,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Required tool '{}' is not available.".format(args[0]),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0]},
        ) from exc

    if proc.returncode != 0:
        raise DiagnosticError(
            "TOOL_FAILED",
            "Command failed: {}".format(shell_join(args)),
            command=command,
            profile=profile,
            resource=resource,
            details={
                "tool": args[0],
                "exit_code": proc.returncode,
                "stdout": truncate_output_tail(proc.stdout),
                "stderr": truncate_output_tail(proc.stderr),
            },
        )
    return {"stdout": proc.stdout, "stderr": proc.stderr}

def command_output(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    command: str,
    profile: Optional[str],
    resource: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    try:
        proc = subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        return proc.stdout.strip()
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Required tool '{}' is not available.".format(args[0]),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0]},
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DiagnosticError(
            "TOOL_FAILED",
            "Command failed: {}".format(shell_join(args)),
            command=command,
            profile=profile,
            resource=resource,
            details={
                "tool": args[0],
                "exit_code": exc.returncode,
                "stdout": truncate_output_tail(exc.stdout),
                "stderr": truncate_output_tail(exc.stderr),
            },
        ) from exc

def run_command(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    command: str,
    profile: Optional[str],
    resource: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = False,
    echo_filter: Optional[Any] = None,
) -> None:
    try:
        if capture_output:
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_capture, tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8"
            ) as stderr_capture:
                proc = subprocess.run(
                    list(args),
                    cwd=None if cwd is None else str(cwd),
                    check=False,
                    env=env,
                    stdout=stdout_capture,
                    stderr=stderr_capture,
                    text=True,
                )
                if proc.returncode != 0:
                    stdout_capture.seek(0, os.SEEK_END)
                    stdout_size = stdout_capture.tell()
                    stdout_capture.seek(max(stdout_size - MAX_DIAGNOSTIC_OUTPUT_CHARS, 0))
                    stdout_tail = stdout_capture.read()

                    stderr_capture.seek(0, os.SEEK_END)
                    stderr_size = stderr_capture.tell()
                    stderr_capture.seek(max(stderr_size - MAX_DIAGNOSTIC_OUTPUT_CHARS, 0))
                    stderr_tail = stderr_capture.read()

                    raise DiagnosticError(
                        "TOOL_FAILED",
                        "Command failed: {}".format(shell_join(args)),
                        command=command,
                        profile=profile,
                        resource=resource,
                        details={
                            "tool": args[0],
                            "exit_code": proc.returncode,
                            "stdout": truncate_output_tail(stdout_tail),
                            "stderr": truncate_output_tail(stderr_tail),
                        },
                        )
        elif echo_filter is not None:
            tail = ""
            proc = subprocess.Popen(
                list(args),
                cwd=None if cwd is None else str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                tail = truncate_output_tail(tail + raw_line)
                line = raw_line.rstrip("\n")
                for console_line in render_console_lines(echo_filter(line), line):
                    print(console_line, file=sys.stderr, flush=True)
            returncode = proc.wait()
            if returncode != 0:
                raise DiagnosticError(
                    "TOOL_FAILED",
                    "Command failed: {}".format(shell_join(args)),
                    command=command,
                    profile=profile,
                    resource=resource,
                    details={
                        "tool": args[0],
                        "exit_code": returncode,
                        "stdout": tail,
                        "stderr": "",
                    },
                )
        else:
            subprocess.run(
                list(args),
                cwd=None if cwd is None else str(cwd),
                check=True,
                env=env,
            )
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Required tool '{}' is not available.".format(args[0]),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0]},
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DiagnosticError(
            "TOOL_FAILED",
            "Command failed: {}".format(shell_join(args)),
            command=command,
            profile=profile,
            resource=resource,
            details={
                "tool": args[0],
                "exit_code": exc.returncode,
                "stdout": truncate_output_tail(exc.stdout),
                "stderr": truncate_output_tail(exc.stderr),
            },
        ) from exc

def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)

def has_numeric_token(text: str) -> bool:
    return bool(re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text))

def resolve_executable_path(executable: str) -> Optional[str]:
    candidate = Path(executable)
    if candidate.is_absolute():
        return str(candidate) if candidate.exists() else None
    return shutil.which(executable)

def first_nonempty_line(*texts: str) -> str:
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return ""

def run_probe_command(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        return {
            "found": False,
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "args": list(args),
        }

    return {
        "found": True,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "args": list(args),
    }
