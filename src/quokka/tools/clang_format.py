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

from quokka.core.constants import CLANG_FORMAT_FILE_EXTENSIONS

from quokka.core.errors import DiagnosticError

from quokka.core.subprocess import run_command

from quokka.project.context import CliContext

def clang_format_files(context: CliContext, files: Sequence[str]) -> List[str]:
    formatter = shutil.which("clang-format")
    if formatter is None:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "clang-format is required but not installed or not available on PATH.",
            command="format",
        )

    formatted_files = [file for file in files if file.endswith(CLANG_FORMAT_FILE_EXTENSIONS)]
    if not formatted_files:
        return []

    run_command(
        [formatter, "-i", "-style=file"] + formatted_files,
        cwd=context.worktree_root,
        command="format",
        profile=None,
        capture_output=context.json_output,
    )
    return formatted_files
