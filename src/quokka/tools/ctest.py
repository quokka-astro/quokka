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

from quokka.core.constants import EXPECTATION_COMMENT_RE, METRIC_KEYWORDS

from quokka.core.errors import DiagnosticError

from quokka.core.subprocess import has_numeric_token

from quokka.core.types import TestSpec

def ctest_lasttest_log_path(build_dir: Path) -> Path:
    return build_dir / "Testing" / "Temporary" / "LastTest.log"

def parse_ctest_lasttest_output(log_path: Path) -> Dict[str, List[str]]:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    outputs: Dict[str, List[str]] = {}
    current_test: Optional[str] = None
    collecting = False
    current_output: List[str] = []
    test_header_re = re.compile(r"^(?:\d+/\d+\s+)?Test:\s+(.+?)\s*$")

    for line in lines:
        match = test_header_re.match(line)
        if match:
            if current_test is not None and collecting:
                outputs[current_test] = current_output[:]
            current_test = match.group(1).strip()
            collecting = False
            current_output = []
            continue
        if line == "Output:":
            collecting = True
            current_output = []
            continue
        if line == "<end of output>":
            if current_test is not None:
                outputs[current_test] = current_output[:]
            collecting = False
            current_output = []
            continue
        if not collecting:
            continue
        if re.fullmatch(r"-{8,}", line):
            continue
        current_output.append(line.rstrip())

    if current_test is not None and collecting:
        outputs[current_test] = current_output[:]
    return outputs

def extract_metric_lines(output_lines: Sequence[str]) -> List[str]:
    selected: List[str] = []
    seen = set()
    for raw_line in output_lines:
        line = raw_line.strip()
        if not line or line in seen:
            continue
        lower = line.lower()
        if lower.startswith(("initial ", "elapsed time", "tinyprofiler", "unused parmparse", "pinned memory", "cpu memory", "name ", "mpi initialized", "amrex ")):
            continue
        if not has_numeric_token(line):
            continue
        if any(keyword in lower for keyword in METRIC_KEYWORDS):
            selected.append(line)
            seen.add(line)
            continue
        if len(selected) >= 5:
            break
    return selected

def observed_metrics_from_lasttest(build_dir: Path, tests: Sequence[TestSpec]) -> List[Dict[str, Any]]:
    outputs = parse_ctest_lasttest_output(ctest_lasttest_log_path(build_dir))
    observed: List[Dict[str, Any]] = []
    for test in tests:
        lines = extract_metric_lines(outputs.get(test.name, []))
        if not lines:
            continue
        observed.append(
            {
                "test": test.name,
                "lines": lines,
                "source": str(ctest_lasttest_log_path(build_dir)),
            }
        )
    return observed
