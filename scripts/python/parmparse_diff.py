#!/usr/bin/env python3
import argparse
import decimal
import re
import shlex
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple


@dataclass
class ParmEntry:
    values: List[object] = field(default_factory=list)
    raw_values: List[str] = field(default_factory=list)
    lines: List[int] = field(default_factory=list)


def read_logical_lines(path: str) -> Iterable[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        buf = ""
        start_line = 0
        for line_num, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not buf:
                start_line = line_num
                buf = line
            else:
                buf = f"{buf} {line.lstrip()}"
            stripped = buf.rstrip()
            if stripped.endswith("\\"):
                buf = stripped[:-1].rstrip()
                continue
            yield start_line, buf
            buf = ""
        if buf:
            yield start_line, buf


def strip_comments(text: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if ch == "\\":
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single and not in_double:
            if ch == "#":
                return text[:i]
            if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
                return text[:i]
        i += 1
    return text


def split_assignment(line: str) -> Tuple[str, str, str]:
    match = re.match(r"^([A-Za-z0-9_.]+)\s*(\+=|=)\s*(.*)$", line)
    if not match:
        return "", "", ""
    return match.group(1), match.group(2), match.group(3)


def parse_tokens(value: str) -> List[str]:
    if not value:
        return []
    try:
        return shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value.split()


def normalize_token(token: str) -> object:
    lower = token.lower()
    if lower in {"true", "false", "t", "f", "yes", "no"}:
        return lower in {"true", "t", "yes"}
    if re.match(r"^[+-]?\d+$", token):
        try:
            return int(token)
        except ValueError:
            return token
    if re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", token):
        try:
            return decimal.Decimal(token)
        except decimal.InvalidOperation:
            return token
    return token


def parse_parmparse(path: str) -> Tuple[dict, List[str]]:
    entries: dict = {}
    warnings: List[str] = []
    for line_num, raw in read_logical_lines(path):
        stripped = strip_comments(raw).strip()
        if not stripped:
            continue
        key, op, value = split_assignment(stripped)
        if not key:
            warnings.append(f"{path}:{line_num} unable to parse: {raw}")
            continue
        tokens = parse_tokens(value)
        normalized = [normalize_token(tok) for tok in tokens]
        entry = entries.get(key, ParmEntry())
        if op == "=":
            entry.values = normalized
            entry.raw_values = tokens
            entry.lines = [line_num]
        elif op == "+=":
            entry.values.extend(normalized)
            entry.raw_values.extend(tokens)
            entry.lines.append(line_num)
        entries[key] = entry
    return entries, warnings


def format_values(values: List[str]) -> str:
    return " ".join(values) if values else "<empty>"


def values_equal(left: ParmEntry, right: ParmEntry) -> bool:
    return left.values == right.values


def diff_entries(left: dict, right: dict) -> List[str]:
    lines: List[str] = []
    for key in sorted(set(left.keys()) | set(right.keys())):
        if key not in right:
            lines.append(f"- {key} = {format_values(left[key].raw_values)}")
            continue
        if key not in left:
            lines.append(f"+ {key} = {format_values(right[key].raw_values)}")
            continue
        if not values_equal(left[key], right[key]):
            left_val = format_values(left[key].raw_values)
            right_val = format_values(right[key].raw_values)
            lines.append(f"~ {key}: {left_val} -> {right_val}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic diff for AMReX ParmParse input files.",
    )
    parser.add_argument("left", help="Left (reference) ParmParse/TOML input file")
    parser.add_argument("right", help="Right (comparison) ParmParse/TOML input file")
    parser.add_argument(
        "--show-equal",
        action="store_true",
        help="Include keys that match after normalization",
    )
    args = parser.parse_args()

    left, left_warnings = parse_parmparse(args.left)
    right, right_warnings = parse_parmparse(args.right)
    diffs = diff_entries(left, right)

    if args.show_equal:
        for key in sorted(set(left.keys()) & set(right.keys())):
            if values_equal(left[key], right[key]):
                diffs.append(f"= {key} = {format_values(left[key].raw_values)}")

    if diffs:
        for line in diffs:
            print(line)
    else:
        print("No semantic differences found.")

    for warning in left_warnings + right_warnings:
        print(f"warning: {warning}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
