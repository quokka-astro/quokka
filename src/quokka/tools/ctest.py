from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

from quokka.core.constants import METRIC_KEYWORDS
from quokka.core.subprocess import has_numeric_token
from quokka.core.types import TestSpec


def ctest_lasttest_log_path(build_dir: Path) -> Path:
    return build_dir / "Testing" / "Temporary" / "LastTest.log"


def parse_ctest_lasttest_output(log_path: Path) -> dict[str, list[str]]:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    outputs: dict[str, list[str]] = {}
    current_test: Optional[str] = None
    collecting = False
    current_output: list[str] = []
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


def extract_metric_lines(output_lines: Sequence[str]) -> list[str]:
    selected: list[str] = []
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


def observed_metrics_from_lasttest(build_dir: Path, tests: Sequence[TestSpec]) -> list[dict[str, Any]]:
    outputs = parse_ctest_lasttest_output(ctest_lasttest_log_path(build_dir))
    observed: list[dict[str, Any]] = []
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
