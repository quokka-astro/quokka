#!/usr/bin/env python3
"""Ensure every Markdown document is reachable via the MkDocs navigation."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


class _SafeLoader(yaml.SafeLoader):
    """Safe loader that treats Python object tags as plain strings."""


def _construct_python_name(loader: yaml.SafeLoader, suffix: str, node):
    return loader.construct_scalar(node)


_SafeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name", _construct_python_name
)


def _gather_nav_paths(items) -> set[str]:
    paths: set[str] = set()
    if items is None:
        return paths
    if isinstance(items, list):
        for item in items:
            paths.update(_gather_nav_paths(item))
    elif isinstance(items, dict):
        for value in items.values():
            paths.update(_gather_nav_paths(value))
    elif isinstance(items, str):
        # MkDocs allows anchors (e.g., foo.md#section); ignore the anchor part.
        path, *_ = items.split('#', 1)
        if path.endswith('.md'):
            paths.add(Path(path).as_posix())
    else:
        raise TypeError(f"Unsupported nav entry type: {type(items)!r}")
    return paths


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "docs" / "mkdocs.yml"
    if not config_path.is_file():
        raise SystemExit(f"MkDocs configuration not found: {config_path}")

    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_SafeLoader)
    docs_dir = config.get("docs_dir", "docs")
    nav_entries = config.get("nav", [])

    docs_root = (config_path.parent / docs_dir).resolve()
    if not docs_root.is_dir():
        raise SystemExit(f"Docs directory not found: {docs_root}")

    reachable = _gather_nav_paths(nav_entries)
    # The site landing page is always reachable even if not in nav.
    reachable.add("index.md")

    all_docs = {
        path.relative_to(docs_root).as_posix()
        for path in docs_root.rglob("*.md")
    }

    missing_from_repo = sorted(reachable - all_docs)
    unreachable = sorted(all_docs - reachable)

    problems = []
    if missing_from_repo:
        problems.append(
            "The MkDocs navigation references missing files:\n" +
            "\n".join(f"  - {name}" for name in missing_from_repo)
        )
    if unreachable:
        problems.append(
            "Markdown files are not reachable via the MkDocs navigation:\n" +
            "\n".join(f"  - {name}" for name in unreachable)
        )

    if problems:
        print("\n\n".join(problems))
        return 1

    print("All Markdown files are reachable via the MkDocs navigation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
