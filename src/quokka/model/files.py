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

from quokka.core.errors import DiagnosticError

from quokka.project.root import is_subpath

from quokka.project.context import CliContext

def resolve_buildtree_binary(context: CliContext, problem: str, command: str) -> Optional[Path]:
    from quokka.model.tests import discover_tests
    from quokka.project.state import artifact_receipt_path, read_json

    profile = context.require_profile(command)
    receipt_path = artifact_receipt_path(profile.build_dir, problem)
    if receipt_path.exists():
        receipt = read_json(receipt_path, command, context.profile_name())
        binary_path = Path(str(receipt.get("binary_path", "")))
        if binary_path.exists():
            return binary_path

    candidate = profile.build_dir / "src" / "problems" / problem / problem
    if candidate.exists():
        return candidate

    matches = list((profile.build_dir / "src" / "problems").glob("*/{}".format(problem)))
    if matches:
        return matches[0]

    tests = discover_tests(profile.build_dir, command, context.profile_name())
    for test in tests:
        if test.command and Path(test.command[0]).name == problem:
            return Path(test.command[0]).resolve()
    return None

def resolve_input_argument(arguments: Sequence[str], working_directory: Optional[Path], worktree_root: Path) -> Optional[Path]:
    if working_directory is None:
        bases = [worktree_root]
    else:
        bases = [working_directory, worktree_root]
    for arg in arguments:
        candidate = Path(arg)
        for base in bases:
            resolved = candidate if candidate.is_absolute() else (base / candidate).resolve()
            if resolved.exists() and resolved.is_file():
                return resolved
    return None

def default_input_for_problem(context: CliContext, problem: str, command: str) -> Optional[Path]:
    from quokka.model.tests import discover_tests

    profile = context.require_profile(command)
    tests = discover_tests(profile.build_dir, command, context.profile_name())
    for test in tests:
        if test.name == problem and test.command:
            resolved = resolve_input_argument(test.command[1:], test.working_directory, context.worktree_root)
            if resolved is not None:
                return resolved

    candidate = context.worktree_root / "inputs" / "{}.toml".format(problem)
    if candidate.exists():
        return candidate.resolve()
    return None

def resolve_run_input(context: CliContext, problem: str, input_arg: Optional[str], command: str) -> Path:
    if input_arg:
        candidate = Path(input_arg).expanduser()
        if not candidate.is_absolute():
            candidate = (context.worktree_root / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise DiagnosticError(
            "INPUT_REQUIRED",
            "Input file '{}' does not exist.".format(input_arg),
            command=command,
            profile=context.profile_name(),
            resource={"kind": "problem", "name": problem},
            details={"input": input_arg},
        )

    resolved = default_input_for_problem(context, problem, command)
    if resolved is not None:
        return resolved

    raise DiagnosticError(
        "INPUT_REQUIRED",
        "Unable to resolve an input file for '{}'.".format(problem),
        command=command,
        profile=context.profile_name(),
        resource={"kind": "problem", "name": problem},
    )

def relative_or_absolute(path: Path, worktree_root: Path) -> str:
    if is_subpath(path, worktree_root):
        return str(path.relative_to(worktree_root))
    return str(path)
