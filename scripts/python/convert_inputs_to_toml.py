#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
INT_RE = re.compile(r"^[+-]?\d+$")
KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
SPECIAL_FLOATS = {"nan", "inf", "-inf", "+inf"}
BOOLS = {"true": True, "false": False}


@dataclass(frozen=True)
class LogicalLine:
    start_line: int
    text: str
    first_indent: str


def merge_continuations(lines: list[str]) -> list[LogicalLine]:
    logical_lines: list[LogicalLine] = []
    buf = ""
    first_indent = ""
    start_line = 1

    for lineno, line in enumerate(lines, start=1):
        line_without_newline = line.rstrip("\n")

        if not buf:
            start_line = lineno
            first_indent = line_without_newline[: len(line_without_newline) - len(line_without_newline.lstrip())]
            current = line_without_newline
        else:
            current = buf + line_without_newline.lstrip()

        if re.search(r"\\\s*$", line_without_newline):
            buf = re.sub(r"\\\s*$", " ", current)
            continue

        logical_lines.append(LogicalLine(start_line=start_line, text=current, first_indent=first_indent))
        buf = ""

    if buf:
        logical_lines.append(LogicalLine(start_line=start_line, text=buf.rstrip(), first_indent=first_indent))

    return logical_lines


def split_inline_comment(text: str) -> tuple[str, str]:
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "#":
            return text[:i].rstrip(), text[i:]
    return text.rstrip(), ""


def toml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def normalize_bool(token: str) -> str:
    lowered = token.lower()
    if lowered not in BOOLS:
        raise ValueError(f"invalid TOML boolean token: {token}")
    return lowered


def normalize_number(token: str) -> str:
    lowered = token.lower()
    if lowered in SPECIAL_FLOATS:
        return lowered
    if INT_RE.fullmatch(token):
        return str(int(token))

    sign = ""
    rest = token
    if rest[0] in "+-":
        sign = rest[0]
        rest = rest[1:]

    if "e" in rest or "E" in rest:
        mantissa, exponent = re.split(r"[eE]", rest, maxsplit=1)
        if mantissa.startswith("."):
            mantissa = "0" + mantissa
        if mantissa.endswith("."):
            mantissa += "0"
        return f"{sign}{mantissa}e{exponent}"

    if rest.startswith("."):
        rest = "0" + rest
    if rest.endswith("."):
        rest += "0"
    return sign + rest


def classify_token(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return "string"
    lowered = token.lower()
    if lowered in BOOLS:
        return "bool"
    if lowered in SPECIAL_FLOATS or NUMBER_RE.fullmatch(token):
        return "number"
    return "string"


def unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def parse_scalar(token: str) -> Any:
    kind = classify_token(token)
    if kind == "string":
        return unquote(token)
    if kind == "bool":
        return BOOLS[token.lower()]
    lowered = token.lower()
    if lowered == "nan":
        return math.nan
    if lowered == "inf" or lowered == "+inf":
        return math.inf
    if lowered == "-inf":
        return -math.inf
    if INT_RE.fullmatch(token):
        return int(token)
    return float(token)


def token_to_toml(token: str) -> str:
    kind = classify_token(token)
    if kind == "string":
        return toml_quote(unquote(token))
    if kind == "bool":
        return normalize_bool(token)
    return normalize_number(token)


def parse_value(raw_value: str, *, source: str, lineno: int) -> tuple[str, Any]:
    value = raw_value.strip()
    if not value:
        raise ValueError(f"{source}:{lineno}: empty assignment is not valid TOML")

    if value.startswith("= "):
        value = value[1:].lstrip()

    tokens = [value] if (len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}) else value.split()
    if not tokens:
        raise ValueError(f"{source}:{lineno}: failed to parse value")

    token_kinds = {classify_token(token) for token in tokens}

    if len(tokens) == 1:
        return token_to_toml(tokens[0]), parse_scalar(tokens[0])

    if token_kinds == {"number"}:
        return "[" + ", ".join(token_to_toml(token) for token in tokens) + "]", [parse_scalar(token) for token in tokens]
    if token_kinds == {"bool"}:
        return "[" + ", ".join(token_to_toml(token) for token in tokens) + "]", [parse_scalar(token) for token in tokens]
    if token_kinds == {"string"}:
        return "[" + ", ".join(token_to_toml(token) for token in tokens) + "]", [parse_scalar(token) for token in tokens]

    raise ValueError(
        f"{source}:{lineno}: mixed legacy value types are not representable in strict TOML: {value}"
    )


def flatten_toml(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            flatten_toml(child_prefix, child, out)
    else:
        out[prefix] = value


def values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected) and math.isnan(actual):
            return True
        return expected == actual
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(values_equal(e, a) for e, a in zip(expected, actual))
    return expected == actual


def convert_file(path: Path) -> None:
    current_text = path.read_text()
    try:
        tomllib.loads(current_text)
        return
    except Exception:
        pass

    original_lines = current_text.splitlines()
    logical_lines = merge_continuations(original_lines)

    expected: dict[str, Any] = {}
    converted_lines: list[str] = []

    for logical_line in logical_lines:
        stripped = logical_line.text.strip()
        if not stripped or stripped.startswith("#"):
            converted_lines.append(logical_line.text.rstrip())
            continue

        if "=" not in logical_line.text:
            raise ValueError(f"{path}:{logical_line.start_line}: expected assignment or comment")

        left, right = logical_line.text.split("=", maxsplit=1)
        key = left.strip()
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"{path}:{logical_line.start_line}: invalid key '{key}' for TOML")

        value_text, comment = split_inline_comment(right)
        toml_value, parsed_value = parse_value(value_text, source=str(path), lineno=logical_line.start_line)
        expected[key] = parsed_value

        new_line = f"{logical_line.first_indent}{key} = {toml_value}"
        if comment:
            new_line += f"  {comment.lstrip()}"
        converted_lines.append(new_line)

    converted_text = "\n".join(converted_lines) + "\n"
    parsed_toml = tomllib.loads(converted_text)
    flattened: dict[str, Any] = {}
    flatten_toml("", parsed_toml, flattened)

    if set(flattened) != set(expected):
        missing = sorted(set(expected) - set(flattened))
        extra = sorted(set(flattened) - set(expected))
        raise ValueError(f"{path}: TOML key mismatch after conversion; missing={missing}, extra={extra}")

    for key, expected_value in expected.items():
        actual_value = flattened[key]
        if not values_equal(expected_value, actual_value):
            raise ValueError(
                f"{path}: TOML value mismatch for {key}: expected {expected_value!r}, got {actual_value!r}"
            )

    path.write_text(converted_text)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        paths = [Path(arg) for arg in argv[1:]]
    else:
        paths = sorted(Path("inputs").glob("*.toml"))

    if not paths:
        print("no .toml input files found", file=sys.stderr)
        return 1

    for path in paths:
        convert_file(path)

    print(f"converted {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
