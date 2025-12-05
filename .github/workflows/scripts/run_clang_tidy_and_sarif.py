#!/usr/bin/env python3
"""Run clang-tidy on a list of files and emit SARIF diagnostics."""

import argparse
import pathlib
import re
import subprocess
import sys
import json
from typing import Dict, List, Optional, Tuple, DefaultDict


DIAG_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+(?P<severity>warning|error|note):\s+(?P<message>.*)\s\[(?P<check>[^\]]+)\]\s*$"
)


def _sarif_level(severity: str) -> str:
    if severity == "error":
        return "error"
    if severity == "warning":
        return "warning"
    return "note"


def _normalize_path(path: pathlib.Path, repo_root: pathlib.Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root)
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _run_clang_tidy(
    files: List[pathlib.Path],
    build_dir: pathlib.Path,
    config_file: pathlib.Path,
    repo_root: pathlib.Path,
    clang_tidy: str,
    header_filter: str,
    log_output: pathlib.Path,
    line_filters: Optional[List[Dict[str, object]]] = None,
) -> Tuple[List[Dict[str, object]], bool]:
    results: List[Dict[str, object]] = []
    log_lines: List[str] = []
    had_failures = False

    for target in files:
        cmd = [
            clang_tidy,
            target.as_posix(),
            "-p",
            build_dir.as_posix(),
            f"--config-file={config_file.as_posix()}",
            f"--header-filter={header_filter}",
            "--use-color=false",
            "--quiet",
            "--warnings-as-errors=*",
            "--extra-arg=-Wno-unknown-warning-option",
            "--extra-arg-before=--driver-mode=g++",
        ]
        if line_filters:
            cmd.append(f"--line-filter={json.dumps(line_filters)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log_lines.append(proc.stdout)
        log_lines.append(proc.stderr)
        if proc.returncode != 0:
            had_failures = True

        last_primary: Optional[Dict[str, object]] = None
        for line in (proc.stdout + proc.stderr).splitlines():
            match = DIAG_RE.match(line.strip())
            if not match:
                continue
            severity = match.group("severity")
            diag = {
                "ruleId": match.group("check"),
                "level": _sarif_level(severity),
                "message": match.group("message"),
                "file": _normalize_path(pathlib.Path(match.group("file")), repo_root),
                "line": int(match.group("line")),
                "column": int(match.group("col")),
                "notes": [],
            }
            if severity in ("warning", "error"):
                results.append(diag)
                last_primary = diag
                had_failures = True
            elif severity == "note" and last_primary is not None:
                last_primary["notes"].append(diag)

    log_output.write_text("\n".join(log_lines), encoding="utf-8")
    return results, had_failures


def _write_sarif(results: List[Dict[str, object]], output_path: pathlib.Path) -> None:
    rules: Dict[str, Dict[str, object]] = {}
    sarif_results: List[Dict[str, object]] = []

    for diag in results:
        rule_id = str(diag["ruleId"])
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "shortDescription": {"text": rule_id},
            }

        sarif_result: Dict[str, object] = {
            "ruleId": rule_id,
            "level": diag["level"],
            "message": {"text": diag["message"]},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": diag["file"]},
                        "region": {"startLine": diag["line"], "startColumn": diag["column"]},
                    }
                }
            ],
        }

        notes = diag.get("notes", [])
        if notes:
            related = []
            for note in notes:
                related.append(
                    {
                        "message": {"text": note["message"]},
                        "physicalLocation": {
                            "artifactLocation": {"uri": note["file"]},
                            "region": {"startLine": note["line"], "startColumn": note["column"]},
                        },
                    }
                )
            sarif_result["relatedLocations"] = related

        sarif_results.append(sarif_result)

    sarif = {
        "version": "2.1.0",
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0-rtm.5.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "clang-tidy",
                        "informationUri": "https://clang.llvm.org/extra/clang-tidy/",
                        "rules": list(rules.values()),
                    }
                },
                "results": sarif_results,
            }
        ],
    }

    output_path.write_text(json.dumps(sarif, indent=2), encoding="utf-8")


def _parse_diff(diff_path: pathlib.Path, repo_root: pathlib.Path) -> Dict[str, List[List[int]]]:
    """Parse a unified diff and return line ranges per file (inclusive)."""
    file_ranges: DefaultDict[str, List[List[int]]] = DefaultDict(list)
    current_file: Optional[str] = None
    hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for raw_line in diff_path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("+++ "):
            path = raw_line[4:].strip()
            if path == "/dev/null" or not path.startswith("b/"):
                current_file = None
                continue
            abs_path = (repo_root / path[2:]).resolve()
            current_file = _normalize_path(abs_path, repo_root)
            continue

        if current_file is None:
            continue

        hunk_match = hunk_re.match(raw_line)
        if hunk_match:
            start = int(hunk_match.group(1))
            length = int(hunk_match.group(2) or "1")
            end = start + max(length, 1) - 1
            file_ranges[current_file].append([start, end])

    return file_ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clang-tidy and emit SARIF.")
    parser.add_argument("--file-list", help="Path to file containing targets for clang-tidy.")
    parser.add_argument("--diff-file", required=True, help="Unified diff file used to restrict diagnostics to changed lines.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory containing compile_commands.json.")
    parser.add_argument("--config-file", required=True, help="Path to .clang-tidy config file.")
    parser.add_argument("--repo-root", default=".", help="Path to repository root.")
    parser.add_argument("--sarif-output", default="clang-tidy.sarif", help="Where to write SARIF output.")
    parser.add_argument("--log-output", default="clang-tidy.log", help="Where to write raw clang-tidy output.")
    parser.add_argument("--flag-output", default="clang-tidy.has_diagnostics", help="Where to record whether diagnostics were found.")
    parser.add_argument("--clang-tidy-binary", default="clang-tidy", help="clang-tidy binary to execute.")
    parser.add_argument("--header-filter", default=".*", help="Header filter regex.")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    file_list_arg = args.file_list
    build_dir = pathlib.Path(args.build_dir).resolve()
    config_file = pathlib.Path(args.config_file).resolve()
    sarif_output = pathlib.Path(args.sarif_output)
    log_output = pathlib.Path(args.log_output)
    flag_output = pathlib.Path(args.flag_output)

    if file_list_arg:
        file_list_path = pathlib.Path(file_list_arg)
        if not file_list_path.exists():
            print(f"File list {file_list_path} not found", file=sys.stderr)
            sys.exit(1)
        targets = [
            pathlib.Path(line.strip())
            for line in file_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        targets = []

    if not targets:
        sarif_output.write_text("", encoding="utf-8")
        flag_output.write_text("false", encoding="utf-8")
        sys.exit(0)

    diff_file = pathlib.Path(args.diff_file)
    if not diff_file.exists():
        print(f"Diff file {diff_file} not found", file=sys.stderr)
        sys.exit(1)
    line_filters_data = _parse_diff(diff_file, repo_root)
    if not line_filters_data:
        sarif_output.write_text("", encoding="utf-8")
        flag_output.write_text("false", encoding="utf-8")
        sys.exit(0)
    line_filters = [{"name": re.escape(path), "lines": ranges} for path, ranges in line_filters_data.items()]

    diagnostics, had_failures = _run_clang_tidy(
        targets,
        build_dir,
        config_file,
        repo_root,
        args.clang_tidy_binary,
        args.header_filter,
        log_output,
        line_filters=line_filters if line_filters else None,
    )

    flag_output.write_text("true" if had_failures else "false", encoding="utf-8")
    _write_sarif(diagnostics, sarif_output)


if __name__ == "__main__":
    main()
