#!/usr/bin/env python3
"""Ensure every mdBook Markdown document is reachable via SUMMARY.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
# Pages that are intentionally not part of the mdBook table of contents and so are
# exempt from the SUMMARY.md reachability check. AGENTS.md files are agent/contributor
# guidance (not rendered book pages), like the special 404/summary pages.
HIDDEN_PAGES = {"404.md", "SUMMARY.md", "AGENTS.md"}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs" / "markdown"
    summary_path = docs_root / "SUMMARY.md"
    if not summary_path.is_file():
        raise SystemExit(f"mdBook summary not found: {summary_path}")

    summary_text = summary_path.read_text(encoding="utf-8")
    referenced: set[str] = set()
    for target in LINK_RE.findall(summary_text):
        if "://" in target:
            continue
        path, *_ = target.split("#", 1)
        if path.endswith(".md"):
            referenced.add(Path(path).as_posix())

    all_docs = {
        path.relative_to(docs_root).as_posix()
        for path in docs_root.rglob("*.md")
        if path.name not in HIDDEN_PAGES
    }

    missing_from_repo = sorted(referenced - all_docs)
    unreachable = sorted(all_docs - referenced)

    problems: list[str] = []
    if missing_from_repo:
        problems.append(
            "The mdBook summary references missing files:\n"
            + "\n".join(f"  - {name}" for name in missing_from_repo)
        )
    if unreachable:
        problems.append(
            "Markdown files are not reachable via SUMMARY.md:\n"
            + "\n".join(f"  - {name}" for name in unreachable)
        )

    if problems:
        print("\n\n".join(problems))
        return 1

    print("All Markdown files are reachable via SUMMARY.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
