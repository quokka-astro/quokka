#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import re

WORKFLOWS_DIRECTORY = pathlib.Path(".github/workflows")
UBUNTU_LABEL_PATTERN = re.compile(r"(?P<prefix>runs-on:\s*ubuntu-)(?P<version>\d+\.\d+)(?P<suffix>-arm)?\b")
MACOS_LABEL_PATTERN = re.compile(r"(?P<prefix>runs-on:\s*macos-)(?P<version>\d+)(?P<suffix>)\b")


def parse_version(version_string: str) -> tuple[int, ...]:
    return tuple(int(_part) for _part in version_string.split("."))


def bump_labels(
    workflow_text: str,
    label_pattern: re.Pattern[str],
    label_name: str,
    target_version: str,
    changes: list[str],
) -> str:
    target_version_parsed = parse_version(target_version)
    def replace_label(label_match: re.Match[str]) -> str:
        current_version = label_match.group("version")
        if parse_version(current_version) >= target_version_parsed:
            return label_match.group(0)
        suffix = label_match.group("suffix") or ""
        changes.append(f"{label_name}-{current_version}{suffix} -> {label_name}-{target_version}{suffix}")
        return f"{label_match.group('prefix')}{target_version}{suffix}"
    return label_pattern.sub(replace_label, workflow_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raise pinned runs-on labels in .github/workflows to newer runner image versions; never lowers a pin.")
    parser.add_argument("--ubuntu-version", help="target Ubuntu version, e.g. 26.04")
    parser.add_argument("--macos-version", help="target macOS major version, e.g. 26")
    args = parser.parse_args()
    for workflow_path in sorted(WORKFLOWS_DIRECTORY.glob("*.yml")):
        original_text = workflow_path.read_text()
        changes: list[str] = []
        updated_text = original_text
        if args.ubuntu_version:
            updated_text = bump_labels(updated_text, UBUNTU_LABEL_PATTERN, "ubuntu", args.ubuntu_version, changes)
        if args.macos_version:
            updated_text = bump_labels(updated_text, MACOS_LABEL_PATTERN, "macos", args.macos_version, changes)
        if updated_text != original_text:
            workflow_path.write_text(updated_text)
            for change in changes:
                print(f"{workflow_path.as_posix()}: {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
