from __future__ import annotations

import contextlib
import datetime as dt
import os
from pathlib import Path

from quokka.core.errors import DiagnosticError


def current_user() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


def current_uid_or_user() -> str:
    uid_getter = getattr(os, "getuid", None)
    if callable(uid_getter):
        return str(uid_getter())
    return current_user()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_path(path: Path) -> Path:
    expanded = path.expanduser()
    try:
        return Path(os.path.realpath(os.fspath(expanded)))
    except OSError:
        return expanded.absolute()


def ensure_runtime_dir_layout(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("locks", "meta", "runs"):
        (runtime_dir / directory).mkdir(parents=True, exist_ok=True)

    probe_path = runtime_dir / ".quokka-write-probe-{}".format(os.getpid())
    try:
        probe_path.write_text("ok\n", encoding="utf-8")
        with (runtime_dir / "state.db").open("a+b"):
            pass
    finally:
        with contextlib.suppress(FileNotFoundError, OSError):
            probe_path.unlink()


def is_subpath(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def find_worktree_from_cwd(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "quokka.toml").is_file():
            return candidate
    return None


def resolve_worktree_root(worktree: str | None) -> Path:
    if worktree is not None:
        return Path(worktree).expanduser().resolve()
    env_root = os.environ.get("QUOKKA_WORKTREE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    discovered = find_worktree_from_cwd(Path.cwd())
    if discovered is not None:
        return discovered
    raise DiagnosticError(
        "USAGE_ERROR",
        "Unable to resolve the Quokka worktree. Use -C /path/to/worktree or activate the worktree first.",
    )
