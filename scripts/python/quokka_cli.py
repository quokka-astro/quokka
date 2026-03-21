#!/usr/bin/env python3

from __future__ import annotations

import argparse
import configparser
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
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


SCHEMA = 1
LOCK_TYPES = ("build", "run")
TIDY_SELECTORS = {"changed", "previous", "origin", "dev"}
FORMAT_SELECTORS = {"changed", "previous", "origin", "dev", "all"}
REGRESSION_META_SECTIONS = {"main", "AMReX", "source"}
SUBMODULE_PATHS = (
    "extern/amrex",
    "extern/AMReX-Hydro",
    "extern/Microphysics",
    "extern/yaml-cpp",
    "extern/turbulence",
)
IGNORED_PROFILE_DEFINES = frozenset({"AMReX_MPI"})
ENV_OVERRIDE_KEYS = (
    "CC",
    "CXX",
    "CMAKE_GENERATOR",
    "CMAKE_PREFIX_PATH",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CUDA_COMPILER",
)
DIAGNOSTIC_CODES = {
    "OK": 0,
    "USAGE_ERROR": 10,
    "UNKNOWN_PROFILE": 11,
    "UNKNOWN_RESOURCE": 12,
    "PROFILE_UNCONFIGURED": 13,
    "INPUT_REQUIRED": 14,
    "TEST_MAPPING_UNSUPPORTED": 15,
    "RUNTIME_DIR_UNSAFE": 16,
    "TIDY_SELECTOR_INVALID": 17,
    "FORMAT_SELECTOR_INVALID": 18,
    "PRE_COMMIT_UNAVAILABLE": 19,
    "RESOURCE_LOCKED": 20,
    "STALE_ARTIFACT": 21,
    "MISSING_ARTIFACT": 22,
    "CONFIGURE_DRIFT": 23,
    "EXECUTOR_UNAVAILABLE": 24,
    "TOOL_FAILED": 25,
    "STATE_CORRUPT": 26,
    "INTERNAL_ERROR": 30,
}


class DiagnosticError(RuntimeError):
    def __init__(
        self,
        diagnostic_id: str,
        message: str,
        *,
        command: Optional[str] = None,
        profile: Optional[str] = None,
        resource: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_id = diagnostic_id
        self.exit_code = DIAGNOSTIC_CODES[diagnostic_id]
        self.command = command
        self.profile = profile
        self.resource = resource
        self.details = details or {}


@dataclasses.dataclass
class CommandResult:
    command: str
    profile: Optional[str]
    resource: Optional[Dict[str, Any]]
    data: Dict[str, Any]
    text: str


@dataclasses.dataclass
class ProfileConfig:
    name: str
    build_dir: Path
    generator: str
    defines: Dict[str, str]
    executor_kind: str
    executor: Dict[str, Any]


@dataclasses.dataclass
class RepoConfig:
    path: Path
    default_profile: str
    profiles: Dict[str, ProfileConfig]
    policy: Dict[str, Any]


@dataclasses.dataclass
class TestSpec:
    name: str
    command: List[str]
    working_directory: Optional[Path]
    source_path: Path


@dataclasses.dataclass
class LockInfo:
    lock_type: str
    lock_path: Path
    metadata_path: Path
    active: bool
    metadata: Optional[Dict[str, Any]]


@dataclasses.dataclass
class LockHandle:
    context: "CliContext"
    lock_type: str
    file_handle: Any
    lock_path: Path
    metadata_path: Path

    def release(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.metadata_path.unlink()
        try:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.file_handle.close()
        self.context.remove_lock_index(self.lock_type)

    def __enter__(self) -> "LockHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()


class CliContext:
    def __init__(self, worktree_root: Path, config: RepoConfig, profile: Optional[ProfileConfig], *, json_output: bool = False) -> None:
        self.worktree_root = worktree_root
        self.config = config
        self.profile = profile
        self.json_output = json_output
        self.hostname = socket.gethostname()
        self.worktree_id = hashlib.sha256((self.hostname + "\n" + str(self.worktree_root)).encode("utf-8")).hexdigest()[:12]
        self._runtime_dir: Optional[Path] = None
        self._db: Optional[sqlite3.Connection] = None

    def profile_name(self) -> Optional[str]:
        return None if self.profile is None else self.profile.name

    def require_profile(self, command: str) -> ProfileConfig:
        if self.profile is None:
            raise DiagnosticError(
                "UNKNOWN_PROFILE",
                "No profile is selected for this command.",
                command=command,
            )
        return self.profile

    def resolve_runtime_dir(self, command: str) -> Path:
        if self._runtime_dir is not None:
            return self._runtime_dir

        env_runtime = os.environ.get("QUOKKA_RUNTIME_DIR")
        runtime_candidates: List[Path] = []
        if env_runtime:
            runtime_candidates.append(Path(env_runtime).expanduser())
        else:
            xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
            if xdg_runtime:
                runtime_candidates.append(Path(xdg_runtime).expanduser() / "quokka")
            runtime_candidates.append(Path("/tmp") / ("quokka-" + current_uid_or_user()))

        worktree_root = canonical_path(self.worktree_root)
        home_dir = canonical_path(Path.home())
        errors: List[Dict[str, str]] = []
        for runtime_dir in runtime_candidates:
            canonical_runtime_dir = canonical_path(runtime_dir)

            if canonical_runtime_dir == worktree_root or is_subpath(canonical_runtime_dir, worktree_root):
                raise DiagnosticError(
                    "RUNTIME_DIR_UNSAFE",
                    "The runtime directory resolves inside the worktree and is not safe for live state.",
                    command=command,
                    profile=self.profile_name(),
                    details={
                        "runtime_dir": str(runtime_dir),
                        "resolved_runtime_dir": str(canonical_runtime_dir),
                        "worktree_root": str(self.worktree_root),
                    },
                )

            if canonical_runtime_dir == home_dir or is_subpath(canonical_runtime_dir, home_dir):
                raise DiagnosticError(
                    "RUNTIME_DIR_UNSAFE",
                    "The runtime directory resolves inside the home directory and may be NFS-backed.",
                    command=command,
                    profile=self.profile_name(),
                    details={
                        "runtime_dir": str(runtime_dir),
                        "resolved_runtime_dir": str(canonical_runtime_dir),
                        "home": str(home_dir),
                    },
                )

            try:
                ensure_runtime_dir_layout(runtime_dir)
                self._runtime_dir = runtime_dir
                return runtime_dir
            except OSError as exc:
                errors.append(
                    {
                        "runtime_dir": str(runtime_dir),
                        "resolved_runtime_dir": str(canonical_runtime_dir),
                        "error": str(exc),
                    }
                )

        raise DiagnosticError(
            "TOOL_FAILED",
            "Unable to create a usable runtime directory.",
            command=command,
            profile=self.profile_name(),
            details={"attempts": errors},
        )

    def db_path(self, command: str) -> Path:
        return self.resolve_runtime_dir(command) / "state.db"

    def open_db(self, command: str) -> sqlite3.Connection:
        if self._db is not None:
            return self._db

        db = sqlite3.connect(str(self.db_path(command)))
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL;")
        db.execute("PRAGMA synchronous=NORMAL;")
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS worktree (
              worktree_id TEXT PRIMARY KEY,
              root_path TEXT NOT NULL,
              hostname TEXT NOT NULL,
              first_seen_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS profile (
              worktree_id TEXT NOT NULL,
              profile_id TEXT NOT NULL,
              build_dir TEXT NOT NULL,
              executor_kind TEXT NOT NULL,
              configure_fingerprint TEXT,
              last_seen_at TEXT NOT NULL,
              PRIMARY KEY (worktree_id, profile_id)
            );
            CREATE TABLE IF NOT EXISTS lock_index (
              worktree_id TEXT NOT NULL,
              lock_type TEXT NOT NULL,
              profile_id TEXT,
              pid INTEGER NOT NULL,
              boot_id TEXT NOT NULL,
              hostname TEXT NOT NULL,
              metadata_path TEXT NOT NULL,
              started_at TEXT NOT NULL,
              PRIMARY KEY (worktree_id, lock_type)
            );
            CREATE TABLE IF NOT EXISTS artifact_index (
              worktree_id TEXT NOT NULL,
              profile_id TEXT NOT NULL,
              artifact_id TEXT NOT NULL,
              receipt_path TEXT NOT NULL,
              source_fingerprint TEXT NOT NULL,
              configure_fingerprint TEXT NOT NULL,
              built_at TEXT NOT NULL,
              PRIMARY KEY (worktree_id, profile_id, artifact_id)
            );
            CREATE TABLE IF NOT EXISTS event_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT NOT NULL,
              worktree_id TEXT NOT NULL,
              profile_id TEXT,
              event_type TEXT NOT NULL,
              details_json TEXT NOT NULL
            );
            """
        )
        db.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema", str(SCHEMA)),
        )
        timestamp = utc_now()
        db.execute(
            """
            INSERT INTO worktree(worktree_id, root_path, hostname, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(worktree_id) DO UPDATE SET
              root_path=excluded.root_path,
              hostname=excluded.hostname,
              last_seen_at=excluded.last_seen_at
            """,
            (self.worktree_id, str(self.worktree_root), self.hostname, timestamp, timestamp),
        )
        if self.profile is not None:
            db.execute(
                """
                INSERT INTO profile(worktree_id, profile_id, build_dir, executor_kind, configure_fingerprint, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(worktree_id, profile_id) DO UPDATE SET
                  build_dir=excluded.build_dir,
                  executor_kind=excluded.executor_kind,
                  configure_fingerprint=excluded.configure_fingerprint,
                  last_seen_at=excluded.last_seen_at
                """,
                (
                    self.worktree_id,
                    self.profile.name,
                    str(self.profile.build_dir),
                    self.profile.executor_kind,
                    None,
                    timestamp,
                ),
            )
        db.commit()
        self._db = db
        return db

    def update_profile_index(self, configure_fingerprint: Optional[str], command: str) -> None:
        if self.profile is None:
            return
        db = self.open_db(command)
        db.execute(
            """
            INSERT INTO profile(worktree_id, profile_id, build_dir, executor_kind, configure_fingerprint, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(worktree_id, profile_id) DO UPDATE SET
              build_dir=excluded.build_dir,
              executor_kind=excluded.executor_kind,
              configure_fingerprint=excluded.configure_fingerprint,
              last_seen_at=excluded.last_seen_at
            """,
            (
                self.worktree_id,
                self.profile.name,
                str(self.profile.build_dir),
                self.profile.executor_kind,
                configure_fingerprint,
                utc_now(),
            ),
        )
        db.commit()

    def update_artifact_index(self, artifact_id: str, receipt: Dict[str, Any], command: str) -> None:
        profile = self.require_profile(command)
        db = self.open_db(command)
        db.execute(
            """
            INSERT INTO artifact_index(worktree_id, profile_id, artifact_id, receipt_path, source_fingerprint, configure_fingerprint, built_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worktree_id, profile_id, artifact_id) DO UPDATE SET
              receipt_path=excluded.receipt_path,
              source_fingerprint=excluded.source_fingerprint,
              configure_fingerprint=excluded.configure_fingerprint,
              built_at=excluded.built_at
            """,
            (
                self.worktree_id,
                profile.name,
                artifact_id,
                str(artifact_receipt_path(profile.build_dir, artifact_id)),
                receipt["source_fingerprint"],
                receipt["configure_fingerprint"],
                receipt["built_at"],
            ),
        )
        db.commit()

    def update_lock_index(self, lock_type: str, metadata: Dict[str, Any], command: str) -> None:
        db = self.open_db(command)
        db.execute(
            """
            INSERT INTO lock_index(worktree_id, lock_type, profile_id, pid, boot_id, hostname, metadata_path, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worktree_id, lock_type) DO UPDATE SET
              profile_id=excluded.profile_id,
              pid=excluded.pid,
              boot_id=excluded.boot_id,
              hostname=excluded.hostname,
              metadata_path=excluded.metadata_path,
              started_at=excluded.started_at
            """,
            (
                self.worktree_id,
                lock_type,
                self.profile_name(),
                metadata["pid"],
                metadata["boot_id"],
                metadata["hostname"],
                metadata["metadata_path"],
                metadata["started_at"],
            ),
        )
        db.commit()

    def remove_lock_index(self, lock_type: str) -> None:
        if self._db is None:
            return
        self._db.execute(
            "DELETE FROM lock_index WHERE worktree_id = ? AND lock_type = ?",
            (self.worktree_id, lock_type),
        )
        self._db.commit()


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


def normalize_define_value(value: Any) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value)


def normalize_profile_defines(raw_defines: Dict[str, Any]) -> Dict[str, str]:
    defines: Dict[str, str] = {}
    for key, value in raw_defines.items():
        if key in IGNORED_PROFILE_DEFINES:
            continue
        defines[key] = normalize_define_value(value)
    return defines


def command_output(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    command: str,
    profile: Optional[str],
    resource: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    try:
        proc = subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        return proc.stdout.strip()
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Required tool '{}' is not available.".format(args[0]),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0]},
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DiagnosticError(
            "TOOL_FAILED",
            "Command failed: {}".format(shell_join(args)),
            command=command,
            profile=profile,
            resource=resource,
            details={
                "tool": args[0],
                "exit_code": exc.returncode,
                "stdout": exc.stdout[-4000:] if exc.stdout else "",
                "stderr": exc.stderr[-4000:] if exc.stderr else "",
            },
        ) from exc


def run_command(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    command: str,
    profile: Optional[str],
    resource: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, str]] = None,
    stdout_to_stderr: bool = False,
) -> None:
    try:
        stdout_stream = sys.stderr if stdout_to_stderr else None
        stderr_stream = sys.stderr if stdout_to_stderr else None
        subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            check=True,
            env=env,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
    except FileNotFoundError as exc:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Required tool '{}' is not available.".format(args[0]),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0]},
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DiagnosticError(
            "TOOL_FAILED",
            "Command failed: {}".format(shell_join(args)),
            command=command,
            profile=profile,
            resource=resource,
            details={"tool": args[0], "exit_code": exc.returncode},
        ) from exc


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def load_repo_config(worktree_root: Path) -> RepoConfig:
    config_path = worktree_root / "quokka.toml"
    if not config_path.exists():
        raise DiagnosticError(
            "STATE_CORRUPT",
            "quokka.toml is missing from the worktree root.",
            details={"config_path": str(config_path)},
        )
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    if data.get("schema") != 1:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "quokka.toml must declare schema = 1.",
            details={"config_path": str(config_path)},
        )

    policy = data.get("policy")
    if not isinstance(policy, dict) or "default_profile" not in policy:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "quokka.toml must define policy.default_profile.",
            details={"config_path": str(config_path)},
        )

    raw_profiles = data.get("profile")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "quokka.toml must define at least one profile.",
            details={"config_path": str(config_path)},
        )

    profiles: Dict[str, ProfileConfig] = {}
    for profile_name, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            raise DiagnosticError(
                "STATE_CORRUPT",
                "Profile '{}' must be a table.".format(profile_name),
                details={"config_path": str(config_path)},
            )
        build_dir_value = raw_profile.get("build_dir")
        executor_value = raw_profile.get("executor")
        if not build_dir_value or not isinstance(build_dir_value, str):
            raise DiagnosticError(
                "STATE_CORRUPT",
                "Profile '{}' is missing build_dir.".format(profile_name),
                details={"config_path": str(config_path)},
            )
        if not isinstance(executor_value, dict) or "kind" not in executor_value:
            raise DiagnosticError(
                "STATE_CORRUPT",
                "Profile '{}' is missing executor.kind.".format(profile_name),
                details={"config_path": str(config_path)},
            )
        raw_defines = raw_profile.get("defines", {})
        if raw_defines is None:
            raw_defines = {}
        if not isinstance(raw_defines, dict):
            raise DiagnosticError(
                "STATE_CORRUPT",
                "Profile '{}' has invalid defines.".format(profile_name),
                details={"config_path": str(config_path)},
            )
        defines = normalize_profile_defines(raw_defines)
        build_dir = (worktree_root / build_dir_value).resolve()
        profiles[profile_name] = ProfileConfig(
            name=profile_name,
            build_dir=build_dir,
            generator=str(raw_profile.get("generator", "Ninja")),
            defines=defines,
            executor_kind=str(executor_value.get("kind")),
            executor=dict(executor_value),
        )

    default_profile = str(policy["default_profile"])
    if default_profile not in profiles:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "The default profile '{}' is not defined.".format(default_profile),
            details={"config_path": str(config_path)},
        )

    return RepoConfig(
        path=config_path,
        default_profile=default_profile,
        profiles=profiles,
        policy=dict(policy),
    )


def find_worktree_from_cwd(start: Path) -> Optional[Path]:
    current = start.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "quokka.toml").is_file():
            return candidate
    return None


def resolve_worktree_root(args: argparse.Namespace) -> Path:
    if args.worktree is not None:
        return Path(args.worktree).expanduser().resolve()
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


def resolve_profile(config: RepoConfig, profile_name: Optional[str], command: str) -> ProfileConfig:
    selected = profile_name or os.environ.get("QUOKKA_PROFILE") or config.default_profile
    if selected not in config.profiles:
        raise DiagnosticError(
            "UNKNOWN_PROFILE",
            "Profile '{}' is not defined in quokka.toml.".format(selected),
            command=command,
            profile=selected,
            details={"known_profiles": sorted(config.profiles)},
        )
    return config.profiles[selected]


def build_state_dir(build_dir: Path) -> Path:
    return build_dir / ".quokka"


def artifact_receipts_dir(build_dir: Path) -> Path:
    return build_state_dir(build_dir) / "artifacts"


def artifact_receipt_path(build_dir: Path, artifact_id: str) -> Path:
    return artifact_receipts_dir(build_dir) / "{}.json".format(artifact_id)


def schema_receipt_path(build_dir: Path) -> Path:
    return build_state_dir(build_dir) / "schema.json"


def profile_receipt_path(build_dir: Path) -> Path:
    return build_state_dir(build_dir) / "profile.json"


def configure_receipt_path(build_dir: Path) -> Path:
    return build_state_dir(build_dir) / "configure-receipt.json"


def ensure_buildtree_state_layout(build_dir: Path) -> None:
    artifact_receipts_dir(build_dir).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, command: str, profile: Optional[str]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "State file '{}' is unreadable.".format(path),
            command=command,
            profile=profile,
            details={"path": str(path)},
        ) from exc


def cmake_version(command: str, profile: Optional[str]) -> str:
    output = command_output(["cmake", "--version"], command=command, profile=profile)
    first_line = output.splitlines()[0] if output else ""
    parts = first_line.split()
    return parts[2] if len(parts) >= 3 else first_line


def resolve_executable_path(executable: str) -> Optional[str]:
    candidate = Path(executable)
    if candidate.is_absolute():
        return str(candidate) if candidate.exists() else None
    return shutil.which(executable)


def first_nonempty_line(*texts: str) -> str:
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def run_probe_command(
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            list(args),
            cwd=None if cwd is None else str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        return {
            "found": False,
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "args": list(args),
        }

    return {
        "found": True,
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "args": list(args),
    }


def tool_probe(executable: str, version_args: Sequence[str], *, label: Optional[str] = None) -> Dict[str, Any]:
    name = label or Path(executable).name
    path = resolve_executable_path(executable)
    if path is None:
        return {"tool": name, "path": None, "status": "missing", "version": None, "detail": "not found"}

    probe = run_probe_command([path] + list(version_args))
    version = first_nonempty_line(probe["stdout"], probe["stderr"])
    return {
        "tool": name,
        "path": path,
        "status": "ok" if probe["ok"] else "error",
        "version": version or None,
        "detail": None if probe["ok"] else first_nonempty_line(probe["stderr"], probe["stdout"]) or "probe failed",
        "exit_code": probe["exit_code"],
    }


def generator_tool_probe(generator: str) -> Dict[str, Any]:
    generator_tools = {
        "Ninja": ("ninja", ["--version"]),
        "Unix Makefiles": ("make", ["--version"]),
        "Xcode": ("xcodebuild", ["-version"]),
    }
    if generator not in generator_tools:
        return {"tool": generator, "path": None, "status": "skip", "version": None, "detail": "no probe for generator"}
    executable, version_args = generator_tools[generator]
    result = tool_probe(executable, version_args, label=generator)
    result["generator"] = generator
    return result


def cmake_cache_path(build_dir: Path) -> Path:
    return build_dir / "CMakeCache.txt"


def normalize_cmake_cache_value(entry_type: str, value: str) -> str:
    if entry_type == "BOOL":
        upper = value.upper()
        if upper in {"1", "ON", "TRUE", "YES", "Y"}:
            return "ON"
        if upper in {"0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND", ""}:
            return "OFF"
        return upper
    return value


def read_cmake_cache(build_dir: Path, command: str, profile: Optional[str]) -> Dict[str, Dict[str, str]]:
    path = cmake_cache_path(build_dir)
    if not path.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' is not configured yet.".format(profile or "<none>"),
            command=command,
            profile=profile,
            details={"build_dir": str(build_dir)},
        )

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "CMake cache '{}' is unreadable.".format(path),
            command=command,
            profile=profile,
            details={"path": str(path)},
        ) from exc

    entries: Dict[str, Dict[str, str]] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        key_type, sep, value = line.partition("=")
        if not sep:
            continue
        key, type_sep, entry_type = key_type.partition(":")
        if not type_sep or not key:
            continue
        entries[key] = {
            "type": entry_type,
            "raw_value": value,
            "value": normalize_cmake_cache_value(entry_type, value),
        }
    return entries


def profile_define_state(profile: ProfileConfig, command: str, profile_name: Optional[str]) -> Dict[str, Any]:
    cache_path = cmake_cache_path(profile.build_dir)
    requested = {key: normalize_define_value(profile.defines[key]) for key in sorted(profile.defines)}
    if not cache_path.exists():
        return {
            "cache_path": str(cache_path),
            "configured": False,
            "requested_defines": requested,
            "effective_defines": {},
            "mismatches": [],
        }

    cache_entries = read_cmake_cache(profile.build_dir, command, profile_name)
    effective: Dict[str, str] = {}
    mismatches: List[Dict[str, Any]] = []
    for key in sorted(requested):
        entry = cache_entries.get(key)
        if entry is None:
            mismatches.append(
                {
                    "key": key,
                    "requested": requested[key],
                    "actual": None,
                    "cache_type": None,
                    "cache_raw_value": None,
                    "reason": "missing",
                }
            )
            continue
        effective[key] = entry["value"]
        if entry["value"] != requested[key]:
            mismatches.append(
                {
                    "key": key,
                    "requested": requested[key],
                    "actual": entry["value"],
                    "cache_type": entry["type"],
                    "cache_raw_value": entry["raw_value"],
                    "reason": "value_mismatch",
                }
            )

    return {
        "cache_path": str(cache_path),
        "configured": True,
        "requested_defines": requested,
        "effective_defines": effective,
        "mismatches": mismatches,
    }


def format_define_mismatch_summary(mismatches: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for mismatch in mismatches:
        actual = mismatch.get("actual")
        if actual is None:
            actual_text = "<missing>"
        else:
            actual_text = str(actual)
        parts.append("{}={} (actual {})".format(mismatch.get("key"), mismatch.get("requested"), actual_text))
    return ", ".join(parts)


def cache_entry_value(entries: Dict[str, Dict[str, str]], *keys: str) -> Optional[str]:
    for key in keys:
        entry = entries.get(key)
        if entry is not None:
            return entry["value"]
    return None


def read_cmake_set_file(path: Path) -> Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    assignments: Dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("set(") or not line.endswith(")"):
            continue
        body = line[4:-1].strip()
        if not body:
            continue
        try:
            parts = shlex.split(body, comments=False, posix=True)
        except ValueError:
            continue
        if len(parts) < 2:
            continue
        assignments[parts[0]] = " ".join(parts[1:])
    return assignments


def compiler_metadata_from_build(build_dir: Path, cache_entries: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Optional[str]]]:
    compiler_info: Dict[str, Dict[str, Optional[str]]] = {}
    for language, key in (("C", "c"), ("CXX", "cxx")):
        info: Dict[str, Optional[str]] = {
            "path": cache_entry_value(cache_entries, "CMAKE_{}_COMPILER".format(language)),
            "id": None,
            "version": None,
            "metadata_path": None,
        }
        for metadata_path in sorted((build_dir / "CMakeFiles").glob("*/CMake{}Compiler.cmake".format(language))):
            assignments = read_cmake_set_file(metadata_path)
            if not assignments:
                continue
            info["metadata_path"] = str(metadata_path)
            info["id"] = assignments.get("CMAKE_{}_COMPILER_ID".format(language))
            info["version"] = assignments.get("CMAKE_{}_COMPILER_VERSION".format(language))
            break
        compiler_info[key] = info
    return compiler_info


def build_summary_from_cache(
    profile: ProfileConfig,
    cache_entries: Dict[str, Dict[str, str]],
    define_state: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    requested = define_state["requested_defines"]
    return {
        "generator": cache_entry_value(cache_entries, "CMAKE_GENERATOR") or profile.generator,
        "build_type": cache_entry_value(cache_entries, "CMAKE_BUILD_TYPE") or requested.get("CMAKE_BUILD_TYPE"),
        "mpi": cache_entry_value(cache_entries, "AMReX_MPI") or requested.get("AMReX_MPI"),
        "space_dim": cache_entry_value(cache_entries, "AMReX_SPACEDIM") or requested.get("AMReX_SPACEDIM"),
        "gpu_backend": cache_entry_value(cache_entries, "AMReX_GPU_BACKEND") or requested.get("AMReX_GPU_BACKEND"),
        "hdf5_dir": cache_entry_value(cache_entries, "HDF5_DIR", "HDF5_ROOT"),
        "hdf5_diff": cache_entry_value(cache_entries, "HDF5_DIFF_EXECUTABLE"),
        "python_enabled": cache_entry_value(cache_entries, "QUOKKA_PYTHON"),
        "python_executable": cache_entry_value(cache_entries, "_Python_EXECUTABLE", "Python_EXECUTABLE"),
    }


def doctor_python_executable(cache_entries: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], str]:
    cached = cache_entry_value(cache_entries, "_Python_EXECUTABLE", "Python_EXECUTABLE")
    if cached:
        return cached, "cache"
    for candidate in ("python3", "python"):
        path = resolve_executable_path(candidate)
        if path is not None:
            return path, "path"
    return None, "missing"


def probe_python_stack(python_executable: Optional[str], source: str) -> Dict[str, Any]:
    if python_executable is None:
        return {
            "status": "missing",
            "executable": None,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
        }

    script = "\n".join(
        [
            "import json",
            "results = {}",
            "checks = [",
            "    ('numpy', 'import numpy'),",
            "    ('matplotlib', 'import matplotlib'),",
            "    ('matplotlib.pyplot', 'import matplotlib.pyplot'),",
            "    ('matplotlib.cm', 'import matplotlib.cm'),",
            "    ('PIL', 'from PIL import Image'),",
            "]",
            "for name, stmt in checks:",
            "    try:",
            "        exec(stmt, {})",
            "        results[name] = {'ok': True}",
            "    except Exception as exc:",
            "        results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}",
            "print(json.dumps(results))",
        ]
    )
    probe = run_probe_command([python_executable, "-c", script])
    if not probe["found"]:
        return {
            "status": "missing",
            "executable": python_executable,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
            "detail": "interpreter not found",
        }
    if not probe["ok"]:
        return {
            "status": "error",
            "executable": python_executable,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
            "detail": first_nonempty_line(probe["stderr"], probe["stdout"]) or "probe failed",
        }

    try:
        modules = json.loads(probe["stdout"]) if probe["stdout"] else {}
    except json.JSONDecodeError:
        return {
            "status": "error",
            "executable": python_executable,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
            "detail": "invalid probe output",
        }

    plotting_modules = ("matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL")
    check_order = ("numpy",) + plotting_modules
    numpy_available = bool((modules.get("numpy") or {}).get("ok"))
    plotting_available = all(bool((modules.get(name) or {}).get("ok")) for name in plotting_modules)
    failed_modules = [name for name in check_order if not bool((modules.get(name) or {}).get("ok"))]
    status = "ok" if plotting_available else ("partial" if numpy_available or any(payload.get("ok") for payload in modules.values()) else "error")
    first_error = ""
    for name in ("numpy",) + plotting_modules:
        payload = modules.get(name) or {}
        if not payload.get("ok"):
            first_error = "{}: {}".format(name, payload.get("error", "probe failed"))
            break

    return {
        "status": status,
        "executable": python_executable,
        "source": source,
        "numpy_available": numpy_available,
        "plotting_available": plotting_available,
        "failed_modules": failed_modules,
        "modules": modules,
        "detail": first_error or None,
    }


def compute_configure_fingerprint(context: CliContext, command: str) -> str:
    profile = context.require_profile(command)
    payload = {
        "profile": profile.name,
        "build_dir": str(profile.build_dir),
        "generator": profile.generator,
        "cmake_path": shutil.which("cmake"),
        "cmake_version": cmake_version(command, context.profile_name()),
        "executor_kind": profile.executor_kind,
        "defines": {key: profile.defines[key] for key in sorted(profile.defines)},
        "env_overrides": {key: os.environ.get(key) for key in ENV_OVERRIDE_KEYS if os.environ.get(key)},
    }
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_status_lines(worktree_root: Path, paths: Sequence[str]) -> List[str]:
    args = ["git", "status", "--porcelain=v1", "--untracked-files=all", "--"] + list(paths)
    output = command_output(args, cwd=worktree_root, command="status", profile=None)
    if not output:
        return []
    return [line for line in output.splitlines() if line]


def git_rev_parse(worktree_root: Path, rev: str) -> str:
    return command_output(["git", "rev-parse", rev], cwd=worktree_root, command="status", profile=None)


def compute_source_fingerprint(worktree_root: Path, input_path: Optional[Path], command: str, profile: Optional[str]) -> str:
    digest = hashlib.sha256()
    head = command_output(["git", "rev-parse", "HEAD"], cwd=worktree_root, command=command, profile=profile)
    digest.update(("HEAD\n" + head + "\n").encode("utf-8"))

    for submodule in SUBMODULE_PATHS:
        submodule_path = worktree_root / submodule
        if (submodule_path / ".git").exists() or submodule_path.exists():
            with contextlib.suppress(DiagnosticError):
                sha = command_output(["git", "-C", str(submodule_path), "rev-parse", "HEAD"], command=command, profile=profile)
                digest.update(("SUBMODULE {} {}\n".format(submodule, sha)).encode("utf-8"))

    pathspecs = ["CMakeLists.txt", "cmake", "src"]
    if input_path is not None and is_subpath(input_path, worktree_root):
        relative_input = str(input_path.relative_to(worktree_root))
        if relative_input not in pathspecs:
            pathspecs.append(relative_input)

    status_lines = git_status_lines(worktree_root, pathspecs)
    for line in status_lines:
        digest.update((line + "\n").encode("utf-8"))
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        candidate = (worktree_root / path_text).resolve()
        if candidate.exists() and candidate.is_file():
            digest.update((path_text + "\0" + file_hash(candidate)).encode("utf-8"))

    if input_path is not None and not is_subpath(input_path, worktree_root):
        digest.update(("EXTERNAL_INPUT {}\n".format(str(input_path))).encode("utf-8"))
        if input_path.exists() and input_path.is_file():
            digest.update(file_hash(input_path).encode("utf-8"))

    return "sha256:" + digest.hexdigest()


def git_metadata(worktree_root: Path, command: str, profile: Optional[str]) -> Dict[str, Any]:
    head = command_output(["git", "rev-parse", "HEAD"], cwd=worktree_root, command=command, profile=profile)
    dirty = bool(command_output(["git", "status", "--porcelain", "-uno"], cwd=worktree_root, command=command, profile=profile))
    submodules: Dict[str, str] = {}
    for submodule in SUBMODULE_PATHS:
        sub_path = worktree_root / submodule
        with contextlib.suppress(DiagnosticError):
            if sub_path.exists():
                submodules[submodule] = command_output(["git", "-C", str(sub_path), "rev-parse", "HEAD"], command=command, profile=profile)
    return {"head": head, "dirty": dirty, "submodules": submodules}


def parse_ctest_testfiles(build_dir: Path, command: str, profile: Optional[str]) -> List[TestSpec]:
    root_testfile = build_dir / "CTestTestfile.cmake"
    if not root_testfile.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' is not configured yet.".format(profile or "<none>"),
            command=command,
            profile=profile,
            details={"build_dir": str(build_dir)},
        )

    tests: List[TestSpec] = []
    pattern_add = re.compile(r"^add_test\((.*)\)$")
    pattern_props = re.compile(r"^set_tests_properties\((.*)\)$")
    pattern_subdirs = re.compile(r'^subdirs\("(.+)"\)$')

    pending = [root_testfile]
    visited = set()

    while pending:
        testfile = pending.pop(0).resolve()
        if testfile in visited:
            continue
        visited.add(testfile)
        by_name: Dict[str, TestSpec] = {}
        try:
            lines = testfile.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DiagnosticError(
                "STATE_CORRUPT",
                "CTest metadata is unreadable.",
                command=command,
                profile=profile,
                details={"path": str(testfile)},
            ) from exc

        for raw_line in lines:
            line = raw_line.strip()
            match_subdirs = pattern_subdirs.match(line)
            if match_subdirs:
                child_testfile = (testfile.parent / match_subdirs.group(1) / "CTestTestfile.cmake").resolve()
                if child_testfile.exists():
                    pending.append(child_testfile)
                continue

            match_add = pattern_add.match(line)
            if match_add:
                parts = shlex.split(match_add.group(1))
                if len(parts) >= 2:
                    spec = TestSpec(
                        name=parts[0],
                        command=parts[1:],
                        working_directory=None,
                        source_path=testfile,
                    )
                    by_name[spec.name] = spec
                    tests.append(spec)
                continue

            match_props = pattern_props.match(line)
            if match_props:
                parts = shlex.split(match_props.group(1))
                if len(parts) < 3:
                    continue
                test_name = parts[0]
                if test_name not in by_name:
                    continue
                if "PROPERTIES" not in parts:
                    continue
                props = parts[parts.index("PROPERTIES") + 1 :]
                for index in range(0, len(props) - 1, 2):
                    if props[index] == "WORKING_DIRECTORY":
                        by_name[test_name].working_directory = Path(props[index + 1]).resolve()

    return tests


def strip_cmake_comments(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        chars: List[str] = []
        in_quote = False
        escaped = False
        for ch in raw_line:
            if ch == '"' and not escaped:
                in_quote = not in_quote
            if ch == "#" and not in_quote:
                break
            chars.append(ch)
            if ch == "\\" and not escaped:
                escaped = True
            else:
                escaped = False
        cleaned_lines.append("".join(chars))
    return "\n".join(cleaned_lines)


def iter_cmake_invocations(text: str) -> Iterator[Tuple[str, str]]:
    source = strip_cmake_comments(text)
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    index = 0
    while index < len(source):
        match = pattern.search(source, index)
        if match is None:
            break
        name = match.group(0).lower()
        cursor = match.end()
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source) or source[cursor] != "(":
            index = match.end()
            continue

        depth = 1
        cursor += 1
        body_start = cursor
        in_quote = False
        escaped = False
        while cursor < len(source) and depth > 0:
            ch = source[cursor]
            if ch == '"' and not escaped:
                in_quote = not in_quote
            elif not in_quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            if ch == "\\" and in_quote and not escaped:
                escaped = True
            else:
                escaped = False
            cursor += 1

        if depth != 0:
            break

        yield name, source[body_start:cursor]
        index = cursor + 1


def split_cmake_arguments(body: str) -> List[str]:
    compact = body.replace("\n", " ")
    try:
        return shlex.split(compact, comments=False, posix=True)
    except ValueError:
        return compact.split()


def expand_cmake_token(token: str, variables: Dict[str, str]) -> str:
    return re.sub(r"\$\{([^}]+)\}", lambda match: variables.get(match.group(1), match.group(0)), token)


def normalize_cmake_boolean(value: str) -> Optional[bool]:
    upper = value.upper()
    if upper in {"1", "ON", "TRUE", "YES", "Y"}:
        return True
    if upper in {"0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND", ""}:
        return False
    return None


def evaluate_source_condition(tokens: Sequence[str], profile: ProfileConfig, variables: Dict[str, str]) -> Optional[bool]:
    expanded = [expand_cmake_token(token, variables) for token in tokens]
    if not expanded:
        return None

    if len(expanded) == 1:
        token = expanded[0]
        value = variables.get(token, profile.defines.get(token, token))
        return normalize_cmake_boolean(str(value))

    lhs_token = expanded[0]
    operator = expanded[1].upper()
    rhs_token = expanded[2] if len(expanded) >= 3 else ""
    lhs = str(variables.get(lhs_token, profile.defines.get(lhs_token, lhs_token)))
    rhs = str(variables.get(rhs_token, profile.defines.get(rhs_token, rhs_token)))

    if operator == "MATCHES":
        try:
            return re.search(rhs, lhs) is not None
        except re.error:
            return None

    if operator in {"EQUAL", "GREATER_EQUAL"}:
        try:
            lhs_num = int(lhs)
            rhs_num = int(rhs)
        except ValueError:
            if operator == "EQUAL":
                return lhs == rhs
            return None
        if operator == "EQUAL":
            return lhs_num == rhs_num
        return lhs_num >= rhs_num

    if operator == "STREQUAL":
        return lhs == rhs

    return None


def source_problem_cmakelists(worktree_root: Path) -> List[Path]:
    problems_root = worktree_root / "src" / "problems"
    if not problems_root.exists():
        raise DiagnosticError(
            "STATE_CORRUPT",
            "Problem source directory is missing.",
            command="list",
            details={"path": str(problems_root)},
        )
    return sorted(path for path in problems_root.glob("*/CMakeLists.txt") if path.is_file())


def resolve_source_working_directory(raw_value: str, worktree_root: Path, source_dir: Path) -> Path:
    resolved = raw_value.replace("${CMAKE_SOURCE_DIR}", str(worktree_root)).replace("${CMAKE_CURRENT_SOURCE_DIR}", str(source_dir))
    path = Path(resolved)
    if not path.is_absolute():
        path = (source_dir / path).resolve()
    return path.resolve()


def source_testspec_from_add_test(
    tokens: Sequence[str],
    variables: Dict[str, str],
    cmake_path: Path,
    worktree_root: Path,
) -> Optional[TestSpec]:
    expanded = [expand_cmake_token(token, variables) for token in tokens]
    if not expanded:
        return None

    if "NAME" not in expanded:
        if len(expanded) < 2:
            return None
        return TestSpec(name=expanded[0], command=list(expanded[1:]), working_directory=None, source_path=cmake_path)

    name_index = expanded.index("NAME")
    if name_index + 1 >= len(expanded):
        return None
    test_name = expanded[name_index + 1]

    command_tokens: List[str] = []
    if "COMMAND" in expanded:
        command_index = expanded.index("COMMAND")
        command_end = len(expanded)
        for keyword in ("WORKING_DIRECTORY", "CONFIGURATIONS", "COMMAND_EXPAND_LISTS"):
            if keyword in expanded[command_index + 1 :]:
                command_end = min(command_end, expanded.index(keyword, command_index + 1))
        command_tokens = list(expanded[command_index + 1 : command_end])

    working_directory: Optional[Path] = None
    if "WORKING_DIRECTORY" in expanded:
        wd_index = expanded.index("WORKING_DIRECTORY")
        if wd_index + 1 < len(expanded):
            working_directory = resolve_source_working_directory(expanded[wd_index + 1], worktree_root, cmake_path.parent)

    return TestSpec(name=test_name, command=command_tokens, working_directory=working_directory, source_path=cmake_path)


def source_testspec_from_quokka_add_problem(
    tokens: Sequence[str],
    variables: Dict[str, str],
    cmake_path: Path,
    worktree_root: Path,
) -> Optional[TestSpec]:
    kwargs: Dict[str, str] = {}
    recognized = {"JOB_NAME", "INPUT_FILE", "ADD_TEST", "TEST_PARAMS", "PRIORITY"}
    index = 0
    while index < len(tokens):
        key = tokens[index].upper()
        if key in recognized and index + 1 < len(tokens):
            kwargs[key] = expand_cmake_token(tokens[index + 1], variables)
            index += 2
            continue
        index += 1

    job_name = kwargs.get("JOB_NAME")
    if not job_name:
        return None
    if kwargs.get("ADD_TEST", "ON").upper() == "OFF":
        return None

    input_file = kwargs.get("INPUT_FILE", "{}.toml".format(job_name))
    return TestSpec(
        name=job_name,
        command=[job_name, "../inputs/{}".format(input_file), "${QuokkaTestParams}"],
        working_directory=(worktree_root / "tests").resolve(),
        source_path=cmake_path,
    )


def parse_source_problem_file(cmake_path: Path, worktree_root: Path, profile: ProfileConfig, command: str) -> Tuple[set[str], List[TestSpec]]:
    try:
        text = cmake_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "Source metadata is unreadable.",
            command=command,
            profile=profile.name,
            details={"path": str(cmake_path)},
        ) from exc

    problems: set[str] = set()
    tests: List[TestSpec] = []
    variables: Dict[str, str] = {}
    active_stack = [True]
    matched_stack: List[bool] = []

    for invocation, body in iter_cmake_invocations(text):
        tokens = split_cmake_arguments(body)

        if invocation == "if":
            condition = evaluate_source_condition(tokens, profile, variables)
            parent_active = active_stack[-1]
            branch_active = parent_active and condition is not False
            active_stack.append(branch_active)
            matched_stack.append(condition is not False)
            continue

        if invocation == "elseif":
            if len(active_stack) == 1 or not matched_stack:
                continue
            parent_active = active_stack[-2]
            already_matched = matched_stack[-1]
            condition = evaluate_source_condition(tokens, profile, variables)
            branch_matches = condition is not False
            active_stack[-1] = parent_active and (not already_matched) and branch_matches
            matched_stack[-1] = already_matched or branch_matches
            continue

        if invocation == "else":
            if len(active_stack) == 1 or not matched_stack:
                continue
            parent_active = active_stack[-2]
            active_stack[-1] = parent_active and (not matched_stack[-1])
            matched_stack[-1] = True
            continue

        if invocation == "endif":
            if len(active_stack) > 1:
                active_stack.pop()
            if matched_stack:
                matched_stack.pop()
            continue

        if not active_stack[-1]:
            continue

        if invocation == "set" and len(tokens) >= 2:
            variables[tokens[0]] = expand_cmake_token(tokens[1], variables)
            continue

        if invocation == "quokka_add_problem":
            spec = source_testspec_from_quokka_add_problem(tokens, variables, cmake_path, worktree_root)
            job_name = None
            for index, token in enumerate(tokens[:-1]):
                if token.upper() == "JOB_NAME":
                    job_name = expand_cmake_token(tokens[index + 1], variables)
                    break
            if job_name:
                problems.add(job_name)
            if spec is not None:
                tests.append(spec)
            continue

        if invocation == "add_executable" and tokens:
            target = expand_cmake_token(tokens[0], variables)
            if target:
                problems.add(target)
            continue

        if invocation == "add_test":
            spec = source_testspec_from_add_test(tokens, variables, cmake_path, worktree_root)
            if spec is not None:
                tests.append(spec)

    return problems, tests


def discover_source_problems(worktree_root: Path, profile: ProfileConfig, command: str) -> List[str]:
    problems: set[str] = set()
    for cmake_path in source_problem_cmakelists(worktree_root):
        problems.add(cmake_path.parent.name)
        file_problems, _ = parse_source_problem_file(cmake_path, worktree_root, profile, command)
        problems.update(file_problems)
    return sorted(problems)


def discover_source_tests(worktree_root: Path, profile: ProfileConfig, command: str) -> List[TestSpec]:
    tests_by_name: Dict[str, TestSpec] = {}
    for cmake_path in source_problem_cmakelists(worktree_root):
        _, file_tests = parse_source_problem_file(cmake_path, worktree_root, profile, command)
        for test in file_tests:
            tests_by_name.setdefault(test.name, test)
    return [tests_by_name[name] for name in sorted(tests_by_name)]


def discover_problems(build_dir: Path, command: str, profile: Optional[str]) -> List[str]:
    root_testfile = build_dir / "CTestTestfile.cmake"
    if not root_testfile.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' is not configured yet.".format(profile or "<none>"),
            command=command,
            profile=profile,
            details={"build_dir": str(build_dir)},
        )

    problems = set()
    pattern_subdirs = re.compile(r'^subdirs\("(.+)"\)$')

    problems_index = build_dir / "src" / "problems" / "CTestTestfile.cmake"
    if problems_index.exists():
        try:
            lines = problems_index.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DiagnosticError(
                "STATE_CORRUPT",
                "CTest metadata is unreadable.",
                command=command,
                profile=profile,
                details={"path": str(problems_index)},
            ) from exc

        for raw_line in lines:
            line = raw_line.strip()
            match_subdirs = pattern_subdirs.match(line)
            if match_subdirs:
                problem_name = Path(match_subdirs.group(1)).name
                if problem_name:
                    problems.add(problem_name)

    tests = parse_ctest_testfiles(build_dir, command, profile)
    for test in tests:
        source_parent = test.source_path.resolve().parent
        if source_parent.parent == problems_index.parent:
            problems.add(source_parent.name)
            continue

        for argument in test.command:
            candidate = Path(argument)
            if candidate.parent == problems_index.parent / candidate.stem:
                problems.add(candidate.stem)
                break

    receipts = artifact_receipts_dir(build_dir)
    if receipts.exists():
        for receipt in receipts.glob("*.json"):
            problems.add(receipt.stem)

    problems_root = build_dir / "src" / "problems"
    if problems_root.exists():
        for candidate in problems_root.glob("*/*"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                problems.add(candidate.name)

    return sorted(problems)


def discover_tests(build_dir: Path, command: str, profile: Optional[str]) -> List[TestSpec]:
    return sorted(parse_ctest_testfiles(build_dir, command, profile), key=lambda spec: spec.name)


def discover_suites(worktree_root: Path) -> Tuple[configparser.ConfigParser, List[str]]:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    ini_path = worktree_root / "regression" / "quokka-tests.ini"
    if not ini_path.exists():
        raise DiagnosticError(
            "STATE_CORRUPT",
            "regression/quokka-tests.ini is missing.",
            command="regression",
            details={"path": str(ini_path)},
        )
    parser.read(str(ini_path))
    suites = [section for section in parser.sections() if section not in REGRESSION_META_SECTIONS]
    return parser, sorted(suites)


def resolve_buildtree_binary(context: CliContext, problem: str, command: str) -> Optional[Path]:
    profile = context.require_profile(command)
    receipt_path = artifact_receipt_path(profile.build_dir, problem)
    if receipt_path.exists():
        receipt = read_json(receipt_path, command, context.profile_name())
        binary_path = Path(str(receipt.get("binary_path", "")))
        if binary_path.exists():
            return binary_path

    candidate = profile.build_dir / "src" / "problems" / problem / problem
    if candidate.exists():
        return candidate

    matches = list((profile.build_dir / "src" / "problems").glob("*/{}".format(problem)))
    if matches:
        return matches[0]

    tests = discover_tests(profile.build_dir, command, context.profile_name())
    for test in tests:
        if test.command and Path(test.command[0]).name == problem:
            return Path(test.command[0]).resolve()
    return None


def resolve_input_argument(arguments: Sequence[str], working_directory: Optional[Path], worktree_root: Path) -> Optional[Path]:
    if working_directory is None:
        bases = [worktree_root]
    else:
        bases = [working_directory, worktree_root]
    for arg in arguments:
        candidate = Path(arg)
        for base in bases:
            resolved = candidate if candidate.is_absolute() else (base / candidate).resolve()
            if resolved.exists() and resolved.is_file():
                return resolved
    return None


def default_input_for_problem(context: CliContext, problem: str, command: str) -> Optional[Path]:
    profile = context.require_profile(command)
    tests = discover_tests(profile.build_dir, command, context.profile_name())
    for test in tests:
        if test.name == problem and test.command:
            resolved = resolve_input_argument(test.command[1:], test.working_directory, context.worktree_root)
            if resolved is not None:
                return resolved

    candidate = context.worktree_root / "inputs" / "{}.toml".format(problem)
    if candidate.exists():
        return candidate.resolve()
    return None


def resolve_run_input(context: CliContext, problem: str, input_arg: Optional[str], command: str) -> Path:
    if input_arg:
        candidate = Path(input_arg).expanduser()
        if not candidate.is_absolute():
            candidate = (context.worktree_root / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise DiagnosticError(
            "INPUT_REQUIRED",
            "Input file '{}' does not exist.".format(input_arg),
            command=command,
            profile=context.profile_name(),
            resource={"kind": "problem", "name": problem},
            details={"input": input_arg},
        )

    resolved = default_input_for_problem(context, problem, command)
    if resolved is not None:
        return resolved

    raise DiagnosticError(
        "INPUT_REQUIRED",
        "Unable to resolve an input file for '{}'.".format(problem),
        command=command,
        profile=context.profile_name(),
        resource={"kind": "problem", "name": problem},
    )


def relative_or_absolute(path: Path, worktree_root: Path) -> str:
    if is_subpath(path, worktree_root):
        return str(path.relative_to(worktree_root))
    return str(path)


def state_for_artifact(context: CliContext, artifact_id: str, command: str, input_path: Optional[Path]) -> Tuple[str, Dict[str, Any]]:
    profile = context.require_profile(command)
    receipt_path = artifact_receipt_path(profile.build_dir, artifact_id)
    if not receipt_path.exists():
        return "missing", {"receipt_path": str(receipt_path)}

    receipt = read_json(receipt_path, command, context.profile_name())
    binary_path = Path(str(receipt.get("binary_path", "")))
    if not binary_path.exists():
        return "missing", {"receipt_path": str(receipt_path), "binary_path": str(binary_path)}

    receipt_configure = receipt.get("configure_fingerprint")
    if not isinstance(receipt_configure, str):
        return "unknown", {"receipt_path": str(receipt_path)}
    current_configure = compute_configure_fingerprint(context, command)
    if current_configure != receipt_configure:
        return "stale_configure", {
            "receipt_path": str(receipt_path),
            "configure_fingerprint_previous": receipt_configure,
            "configure_fingerprint_current": current_configure,
        }

    define_state = profile_define_state(profile, command, context.profile_name())
    if define_state["mismatches"]:
        return "stale_configure", {
            "receipt_path": str(receipt_path),
            "cache_path": define_state["cache_path"],
            "requested_defines": define_state["requested_defines"],
            "effective_defines": define_state["effective_defines"],
            "define_mismatches": define_state["mismatches"],
        }

    effective_input = input_path
    if effective_input is None:
        default_input_value = ((receipt.get("inputs") or {}).get("default_input")) if isinstance(receipt.get("inputs"), dict) else None
        if default_input_value:
            effective_input = (context.worktree_root / str(default_input_value)).resolve()

    receipt_source = receipt.get("source_fingerprint")
    if not isinstance(receipt_source, str):
        return "unknown", {"receipt_path": str(receipt_path)}
    current_source = compute_source_fingerprint(context.worktree_root, effective_input, command, context.profile_name())
    if current_source != receipt_source:
        return "stale_source", {
            "receipt_path": str(receipt_path),
            "source_fingerprint_previous": receipt_source,
            "source_fingerprint_current": current_source,
        }

    return "ready", {"receipt": receipt, "binary_path": str(binary_path)}


def ensure_artifact_ready(
    context: CliContext,
    artifact_id: str,
    command: str,
    input_path: Optional[Path],
    build_if_needed: bool,
) -> Dict[str, Any]:
    state, details = state_for_artifact(context, artifact_id, command, input_path)
    if state == "ready":
        return details

    if build_if_needed:
        perform_build(context, [artifact_id], reconfigure=False)
        state, details = state_for_artifact(context, artifact_id, command, input_path)
        if state == "ready":
            return details

    resource = {"kind": "problem", "name": artifact_id}
    if state == "missing":
        raise DiagnosticError(
            "MISSING_ARTIFACT",
            "{} in profile {} is missing and must be built first.".format(artifact_id, context.profile_name()),
            command=command,
            profile=context.profile_name(),
            resource=resource,
            details=details,
        )
    if state == "stale_configure":
        raise DiagnosticError(
            "CONFIGURE_DRIFT",
            "{} in profile {} no longer matches the active build configuration.".format(artifact_id, context.profile_name()),
            command=command,
            profile=context.profile_name(),
            resource=resource,
            details=details,
        )
    if state == "stale_source":
        raise DiagnosticError(
            "STALE_ARTIFACT",
            "{} in profile {} is stale and must be rebuilt before it can run.".format(artifact_id, context.profile_name()),
            command=command,
            profile=context.profile_name(),
            resource=resource,
            details=details,
        )
    raise DiagnosticError(
        "STATE_CORRUPT",
        "{} has unreadable or inconsistent state.".format(artifact_id),
        command=command,
        profile=context.profile_name(),
        resource=resource,
        details=details,
    )


def write_schema_receipt(profile: ProfileConfig) -> None:
    write_json(schema_receipt_path(profile.build_dir), {"schema": 1, "kind": "quokka-buildtree-state"})


def write_profile_receipt(context: CliContext, command: str) -> None:
    profile = context.require_profile(command)
    payload = {
        "schema": 1,
        "profile": profile.name,
        "worktree_root": str(context.worktree_root),
        "build_dir": str(profile.build_dir),
        "generator": profile.generator,
        "executor": profile.executor,
        "defines": profile.defines,
    }
    write_json(profile_receipt_path(profile.build_dir), payload)


def write_configure_receipt(context: CliContext, command: str, define_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = context.require_profile(command)
    if define_state is None:
        define_state = profile_define_state(profile, command, context.profile_name())
    cache_entries = read_cmake_cache(profile.build_dir, command, context.profile_name())
    build_summary = build_summary_from_cache(profile, cache_entries, define_state)
    compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries)
    payload = {
        "schema": 1,
        "configured_at": utc_now(),
        "configure_fingerprint": compute_configure_fingerprint(context, command),
        "cmake_version": cmake_version(command, context.profile_name()),
        "compiler": compiler_info,
        "build": build_summary,
        "generator": profile.generator,
        "source_root": str(context.worktree_root),
        "build_dir": str(profile.build_dir),
        "profile": profile.name,
        "defines": profile.defines,
        "cache_path": define_state["cache_path"],
        "effective_defines": define_state["effective_defines"],
    }
    write_json(configure_receipt_path(profile.build_dir), payload)
    context.update_profile_index(payload["configure_fingerprint"], command)
    return payload


def test_map_by_name(context: CliContext, command: str) -> Dict[str, TestSpec]:
    profile = context.require_profile(command)
    return {test.name: test for test in discover_tests(profile.build_dir, command, context.profile_name())}


def artifact_receipt_payload(
    context: CliContext,
    artifact_id: str,
    binary_path: Path,
    command: str,
) -> Dict[str, Any]:
    default_input = default_input_for_problem(context, artifact_id, command)
    tests = test_map_by_name(context, command)
    test = tests.get(artifact_id)
    working_dir: Optional[Path] = None if test is None else test.working_directory
    source_fingerprint = compute_source_fingerprint(context.worktree_root, default_input, command, context.profile_name())
    payload = {
        "schema": 1,
        "artifact_id": artifact_id,
        "artifact_kind": "problem",
        "profile": context.profile_name(),
        "binary_path": str(binary_path),
        "built_at": utc_now(),
        "source_fingerprint": source_fingerprint,
        "configure_fingerprint": compute_configure_fingerprint(context, command),
        "git": git_metadata(context.worktree_root, command, context.profile_name()),
        "inputs": {
            "default_input": None if default_input is None else relative_or_absolute(default_input, context.worktree_root),
            "default_working_dir": None if working_dir is None else relative_or_absolute(working_dir, context.worktree_root),
        },
    }
    return payload


def write_artifact_receipt(context: CliContext, artifact_id: str, binary_path: Path, command: str) -> Dict[str, Any]:
    payload = artifact_receipt_payload(context, artifact_id, binary_path, command)
    path = artifact_receipt_path(context.require_profile(command).build_dir, artifact_id)
    write_json(path, payload)
    context.update_artifact_index(artifact_id, payload, command)
    return payload


def build_lock_paths(context: CliContext, lock_type: str, command: str) -> Tuple[Path, Path]:
    runtime_dir = context.resolve_runtime_dir(command)
    lock_path = runtime_dir / "locks" / "wt-{}.{}.lock".format(context.worktree_id, lock_type)
    metadata_path = runtime_dir / "meta" / "wt-{}.{}.json".format(context.worktree_id, lock_type)
    return lock_path, metadata_path


def current_boot_id() -> str:
    proc_boot = Path("/proc/sys/kernel/random/boot_id")
    if proc_boot.exists():
        return proc_boot.read_text(encoding="utf-8").strip()
    with contextlib.suppress(Exception):
        return command_output(["sysctl", "-n", "kern.bootsessionuuid"], command="status", profile=None)
    return ""


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def inspect_lock(context: CliContext, lock_type: str, command: str) -> LockInfo:
    lock_path, metadata_path = build_lock_paths(context, lock_type, command)
    metadata = None
    if metadata_path.exists():
        metadata = read_json(metadata_path, command, context.profile_name())
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            active = True
    finally:
        handle.close()
    return LockInfo(lock_type=lock_type, lock_path=lock_path, metadata_path=metadata_path, active=active, metadata=metadata)


def ensure_no_conflicting_locks(context: CliContext, requested: Iterable[str], command: str) -> None:
    for lock_type in requested:
        info = inspect_lock(context, lock_type, command)
        if info.active:
            raise DiagnosticError(
                "RESOURCE_LOCKED",
                "A {} lock is active for this worktree.".format(lock_type),
                command=command,
                profile=context.profile_name(),
                details={
                    "lock_type": lock_type,
                    "metadata_path": str(info.metadata_path),
                    "metadata": info.metadata,
                },
            )


def acquire_lock(context: CliContext, lock_type: str, command: str) -> LockHandle:
    lock_path, metadata_path = build_lock_paths(context, lock_type, command)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        metadata = None
        if metadata_path.exists():
            metadata = read_json(metadata_path, command, context.profile_name())
        raise DiagnosticError(
            "RESOURCE_LOCKED",
            "A {} lock is active for this worktree.".format(lock_type),
            command=command,
            profile=context.profile_name(),
            details={"lock_type": lock_type, "metadata": metadata, "metadata_path": str(metadata_path)},
        ) from exc

    metadata = {
        "schema": 1,
        "lock_type": lock_type,
        "worktree_id": context.worktree_id,
        "worktree_root": str(context.worktree_root),
        "profile": context.profile_name(),
        "pid": os.getpid(),
        "boot_id": current_boot_id(),
        "hostname": context.hostname,
        "command": sys.argv,
        "started_at": utc_now(),
        "metadata_path": str(metadata_path),
    }
    write_json(metadata_path, metadata)
    context.update_lock_index(lock_type, metadata, command)
    return LockHandle(
        context=context,
        lock_type=lock_type,
        file_handle=handle,
        lock_path=lock_path,
        metadata_path=metadata_path,
    )


def break_locks(context: CliContext, command: str) -> List[str]:
    broken: List[str] = []
    for lock_type in LOCK_TYPES:
        info = inspect_lock(context, lock_type, command)
        if not info.metadata_path.exists() and not info.lock_path.exists():
            continue
        if info.active:
            metadata = info.metadata or {}
            pid = metadata.get("pid")
            boot_id = str(metadata.get("boot_id", ""))
            current = current_boot_id()
            if isinstance(pid, int) and pid_is_alive(pid) and (not current or boot_id == current):
                raise DiagnosticError(
                    "RESOURCE_LOCKED",
                    "Cannot break the live {} lock.".format(lock_type),
                    command=command,
                    profile=context.profile_name(),
                    details={"lock_type": lock_type, "metadata": metadata},
                )
        with contextlib.suppress(FileNotFoundError):
            info.metadata_path.unlink()
        with contextlib.suppress(FileNotFoundError):
            info.lock_path.unlink()
        context.remove_lock_index(lock_type)
        broken.append(lock_type)
    return broken


def is_build_configured(build_dir: Path) -> bool:
    return cmake_cache_path(build_dir).exists()


def maybe_reconfigure(context: CliContext, command: str, reconfigure: bool) -> Dict[str, Any]:
    profile = context.require_profile(command)
    ensure_buildtree_state_layout(profile.build_dir)
    write_schema_receipt(profile)
    write_profile_receipt(context, command)

    configure_receipt = configure_receipt_path(profile.build_dir)
    current_fingerprint = compute_configure_fingerprint(context, command)
    needs_configure = reconfigure or not is_build_configured(profile.build_dir)

    if configure_receipt.exists() and not needs_configure:
        receipt = read_json(configure_receipt, command, context.profile_name())
        if receipt.get("configure_fingerprint") != current_fingerprint:
            needs_configure = True
    elif not configure_receipt.exists():
        needs_configure = True

    if needs_configure:
        args = ["cmake", "-S", str(context.worktree_root), "-B", str(profile.build_dir), "-G", profile.generator]
        for key in sorted(profile.defines):
            args.append("-D{}={}".format(key, profile.defines[key]))
        run_command(args, command=command, profile=context.profile_name(), stdout_to_stderr=context.json_output)
    define_state = profile_define_state(profile, command, context.profile_name())
    if define_state["mismatches"]:
        raise DiagnosticError(
            "CONFIGURE_DRIFT",
            "Profile '{}' requested defines do not match the configured CMake cache: {}.".format(
                context.profile_name(), format_define_mismatch_summary(define_state["mismatches"])
            ),
            command=command,
            profile=context.profile_name(),
            details=define_state,
        )
    return write_configure_receipt(context, command, define_state=define_state)


def ensure_profile_configured(context: CliContext, command: str) -> Dict[str, Any]:
    profile = context.require_profile(command)
    if is_build_configured(profile.build_dir):
        return profile_define_state(profile, command, context.profile_name())

    with acquire_lock(context, "build", command):
        return maybe_reconfigure(context, command, reconfigure=False)


def perform_build(context: CliContext, targets: Sequence[str], reconfigure: bool) -> Dict[str, Any]:
    command = "build"
    profile = context.require_profile(command)
    context.resolve_runtime_dir(command)
    context.open_db(command)

    with acquire_lock(context, "build", command):
        configure_receipt = maybe_reconfigure(context, command, reconfigure)
        build_args = ["cmake", "--build", str(profile.build_dir)]
        if targets:
            build_args.extend(["--target"] + list(targets))
        run_command(build_args, command=command, profile=context.profile_name(), stdout_to_stderr=context.json_output)

        if targets:
            requested = list(dict.fromkeys(targets))
        else:
            requested = discover_problems(profile.build_dir, command, context.profile_name())

        receipts_written: List[str] = []
        for artifact_id in requested:
            binary_path = resolve_buildtree_binary(context, artifact_id, command)
            if binary_path is None or not binary_path.exists():
                continue
            write_artifact_receipt(context, artifact_id, binary_path, command)
            receipts_written.append(artifact_id)

    return {
        "build_dir": str(profile.build_dir),
        "configure_fingerprint": configure_receipt["configure_fingerprint"],
        "targets": list(targets),
        "receipts_written": receipts_written,
    }


def git_changed_files(worktree_root: Path, selector: str, command: str, profile: Optional[str]) -> List[str]:
    if selector == "changed":
        output = command_output(["git", "diff", "--name-only", "HEAD"], cwd=worktree_root, command=command, profile=profile)
    elif selector == "previous":
        output = command_output(["git", "diff", "--name-only", "HEAD^"], cwd=worktree_root, command=command, profile=profile)
    elif selector == "origin":
        branch = command_output(["git", "branch", "--show-current"], cwd=worktree_root, command=command, profile=profile)
        output = command_output(
            ["git", "diff", "--name-only", "origin/{}".format(branch)],
            cwd=worktree_root,
            command=command,
            profile=profile,
        )
    elif selector == "dev":
        output = command_output(["git", "diff", "--name-only", "development"], cwd=worktree_root, command=command, profile=profile)
    else:
        raise ValueError(selector)
    return [line for line in output.splitlines() if line]


def ctest_selection(context: CliContext, args: argparse.Namespace) -> List[TestSpec]:
    profile = context.require_profile("test")
    tests = discover_tests(profile.build_dir, "test", context.profile_name())
    if args.ctest_regex:
        pattern = re.compile(args.ctest_regex)
        matched = [test for test in tests if pattern.search(test.name)]
    elif args.test_name:
        matched = [test for test in tests if test.name == args.test_name]
    else:
        matched = tests
    if not matched:
        raise DiagnosticError(
            "UNKNOWN_RESOURCE",
            "No tests matched the requested selector.",
            command="test",
            profile=context.profile_name(),
            resource={"kind": "test", "name": args.test_name or args.ctest_regex or "*"},
        )
    return matched


def problem_for_test(test: TestSpec) -> str:
    if not test.command:
        raise DiagnosticError(
            "TEST_MAPPING_UNSUPPORTED",
            "Test '{}' does not declare a runnable command.".format(test.name),
            command="test",
            resource={"kind": "test", "name": test.name},
        )
    executable = Path(test.command[0]).name
    if not executable:
        raise DiagnosticError(
            "TEST_MAPPING_UNSUPPORTED",
            "Test '{}' cannot be mapped to a single executable.".format(test.name),
            command="test",
            resource={"kind": "test", "name": test.name},
        )
    return executable


def selected_regression_suites(context: CliContext, requested: Sequence[str]) -> Tuple[configparser.ConfigParser, List[str]]:
    parser, suites = discover_suites(context.worktree_root)
    if not requested:
        return parser, suites
    missing = [suite for suite in requested if suite not in suites]
    if missing:
        raise DiagnosticError(
            "UNKNOWN_RESOURCE",
            "Unknown regression suite(s): {}.".format(", ".join(missing)),
            command="regression",
            profile=context.profile_name(),
            resource={"kind": "suite", "name": missing[0]},
            details={"known_suites": suites},
        )
    return parser, list(requested)


def write_subset_regression_ini(parser: configparser.ConfigParser, suites: Sequence[str], worktree_root: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="quokka-regression-", suffix=".ini", dir=str(worktree_root / ".git"), delete=False)
    temp_path = Path(handle.name)
    handle.close()

    subset = configparser.ConfigParser()
    subset.optionxform = str
    for section in parser.sections():
        if section in REGRESSION_META_SECTIONS or section in suites:
            subset[section] = dict(parser[section])
    with temp_path.open("w", encoding="utf-8") as output:
        subset.write(output)
    return temp_path


def format_result(result: CommandResult, as_json: bool) -> str:
    if not as_json:
        return result.text
    payload = {
        "schema": 1,
        "ok": True,
        "command": result.command,
        "profile": result.profile,
        "resource": result.resource,
        "diagnostic": None,
        "data": result.data,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def doctor_hint_command(profile: Optional[str], topic: Optional[str] = None) -> Optional[str]:
    if not profile:
        return None
    args = ["quokka", "doctor"]
    if topic is not None:
        args.append(topic)
    args.extend(["--profile", profile])
    return shell_join(args)


def stream_test_hint_command(profile: Optional[str], resource: Optional[Dict[str, Any]]) -> Optional[str]:
    if not profile:
        return None
    args = ["quokka", "test"]
    selector = None if resource is None else resource.get("selector")
    resource_name = None if resource is None else resource.get("name")
    if selector == "name" and isinstance(resource_name, str) and resource_name != "*":
        args.append(resource_name)
    elif selector == "regex" and isinstance(resource_name, str) and resource_name != "*":
        args.extend(["--ctest-regex", resource_name])
    args.extend(["--profile", profile, "--stream"])
    return shell_join(args)


def diagnostic_hints(error: DiagnosticError, command: Optional[str], profile: Optional[str]) -> List[str]:
    effective_command = error.command or command
    effective_profile = error.profile or profile
    hints: List[str] = []

    doctor_command = None
    if error.diagnostic_id == "RESOURCE_LOCKED":
        doctor_command = doctor_hint_command(effective_profile, "locking")
    elif error.diagnostic_id in {"CONFIGURE_DRIFT", "PROFILE_UNCONFIGURED", "MISSING_ARTIFACT", "STALE_ARTIFACT"}:
        doctor_command = doctor_hint_command(effective_profile, "profile")
    elif error.diagnostic_id in {"TOOL_FAILED", "EXECUTOR_UNAVAILABLE", "STATE_CORRUPT"}:
        doctor_command = doctor_hint_command(effective_profile)

    if doctor_command is not None:
        hints.append("Inspect the current environment with: {}".format(doctor_command))

    if effective_command == "test" and error.diagnostic_id == "TOOL_FAILED":
        stream_command = stream_test_hint_command(effective_profile, error.resource)
        if stream_command is not None:
            hints.append("For live CTest output, rerun with: {}".format(stream_command))

    return hints


def error_result(error: DiagnosticError, as_json: bool, command: Optional[str], profile: Optional[str]) -> str:
    hints = diagnostic_hints(error, command, profile)
    if not as_json:
        if not hints:
            return error.args[0]
        return "{}\nHints:\n- {}".format(error.args[0], "\n- ".join(hints))
    payload = {
        "schema": 1,
        "ok": False,
        "command": error.command or command,
        "profile": error.profile or profile,
        "resource": error.resource,
        "diagnostic": {
            "id": error.diagnostic_id,
            "exit_code": error.exit_code,
            "message": error.args[0],
            "details": error.details,
            "hints": hints,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def cmd_build(context: CliContext, args: argparse.Namespace) -> CommandResult:
    data = perform_build(context, args.targets, reconfigure=args.reconfigure)
    text_targets = ", ".join(data["receipts_written"]) if data["receipts_written"] else "(no problem receipts updated)"
    text = "Built profile {} in {}.\nReceipts updated: {}".format(context.profile_name(), data["build_dir"], text_targets)
    return CommandResult("build", context.profile_name(), None, data, text)


def cmd_run(context: CliContext, args: argparse.Namespace) -> CommandResult:
    profile = context.require_profile("run")
    resource = {"kind": "problem", "name": args.problem}
    context.resolve_runtime_dir("run")
    context.open_db("run")
    ensure_no_conflicting_locks(context, ("build", "run"), "run")
    if args.build_if_needed:
        ensure_profile_configured(context, "run")
    input_path = resolve_run_input(context, args.problem, args.input, "run")
    readiness = ensure_artifact_ready(context, args.problem, "run", input_path, args.build_if_needed)
    binary_path = Path(readiness["binary_path"])

    with acquire_lock(context, "run", "run"):
        working_dir_value = ((readiness.get("receipt") or {}).get("inputs") or {}).get("default_working_dir")
        if working_dir_value:
            working_dir = (context.worktree_root / str(working_dir_value)).resolve()
        else:
            working_dir = context.worktree_root / "tests"
        run_command(
            [str(binary_path), str(input_path)],
            cwd=working_dir,
            command="run",
            profile=context.profile_name(),
            resource=resource,
            stdout_to_stderr=context.json_output,
        )

    data = {
        "binary_path": str(binary_path),
        "input": relative_or_absolute(input_path, context.worktree_root),
        "working_dir": str(working_dir),
    }
    text = "Ran {} with {}.".format(args.problem, data["input"])
    return CommandResult("run", context.profile_name(), resource, data, text)


def cmd_test(context: CliContext, args: argparse.Namespace) -> CommandResult:
    profile = context.require_profile("test")
    if args.stream and args.json:
        raise DiagnosticError(
            "USAGE_ERROR",
            "--stream cannot be combined with --json.",
            command="test",
            profile=context.profile_name(),
        )
    context.resolve_runtime_dir("test")
    context.open_db("test")
    ensure_no_conflicting_locks(context, ("build", "run"), "test")
    if args.build_if_needed:
        ensure_profile_configured(context, "test")
    tests = ctest_selection(context, args)

    unique_targets: List[str] = []
    target_inputs: Dict[str, Optional[Path]] = {}
    for test in tests:
        problem = problem_for_test(test)
        if problem not in unique_targets:
            unique_targets.append(problem)
        target_inputs[problem] = resolve_input_argument(test.command[1:], test.working_directory, context.worktree_root)

    if args.build_if_needed:
        perform_build(context, unique_targets, reconfigure=False)

    for target in unique_targets:
        ensure_artifact_ready(context, target, "test", target_inputs.get(target), build_if_needed=False)

    ctest_args = ["ctest", "--test-dir", str(profile.build_dir)]
    if args.stream:
        ctest_args.extend(["--progress", "--verbose"])
    else:
        ctest_args.append("--output-on-failure")
    if args.ctest_regex:
        ctest_args.extend(["-R", args.ctest_regex])
        resource_name = args.ctest_regex
        resource_selector = "regex"
    elif args.test_name:
        ctest_args.extend(["-R", "^{}$".format(re.escape(args.test_name))])
        resource_name = args.test_name
        resource_selector = "name"
    else:
        resource_name = "*"
        resource_selector = "all"

    test_resource = {"kind": "test", "name": resource_name, "selector": resource_selector}

    with acquire_lock(context, "run", "test"):
        run_command(
            ctest_args,
            command="test",
            profile=context.profile_name(),
            resource=test_resource,
            stdout_to_stderr=context.json_output,
        )

    data = {
        "selected_tests": [test.name for test in tests],
        "build_dir": str(profile.build_dir),
        "streaming": bool(args.stream),
    }
    text = "Ran {} test(s) in profile {}{}.".format(
        len(tests),
        context.profile_name(),
        " with streaming output" if args.stream else "",
    )
    return CommandResult("test", context.profile_name(), test_resource, data, text)


def cmd_regression(context: CliContext, args: argparse.Namespace) -> CommandResult:
    profile = context.require_profile("regression")
    context.resolve_runtime_dir("regression")
    context.open_db("regression")
    ensure_no_conflicting_locks(context, ("build", "run"), "regression")

    parser, suites = selected_regression_suites(context, args.suites)
    targets: List[str] = []
    target_to_input: Dict[str, Optional[Path]] = {}
    for suite in suites:
        target = parser.get(suite, "target", fallback="")
        if not target:
            raise DiagnosticError(
                "STATE_CORRUPT",
                "Regression suite '{}' does not define a target.".format(suite),
                command="regression",
                profile=context.profile_name(),
                resource={"kind": "suite", "name": suite},
            )
        input_file = parser.get(suite, "inputFile", fallback="")
        input_path = None
        if input_file:
            input_path = (context.worktree_root / input_file).resolve()
        if target not in targets:
            targets.append(target)
        target_to_input[target] = input_path

    if args.build_if_needed:
        perform_build(context, targets, reconfigure=False)

    for target in targets:
        ensure_artifact_ready(context, target, "regression", target_to_input.get(target), build_if_needed=False)

    if profile.executor_kind != "local":
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Regression executor kind '{}' is not implemented in the prototype.".format(profile.executor_kind),
            command="regression",
            profile=context.profile_name(),
            details={"executor": profile.executor},
        )

    regtest = context.worktree_root / "extern" / "regression_testing" / "regtest.py"
    if not regtest.exists():
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "Regression harness is unavailable because extern/regression_testing/regtest.py is missing.",
            command="regression",
            profile=context.profile_name(),
            details={"path": str(regtest)},
        )

    temp_ini: Optional[Path] = None
    ini_path = context.worktree_root / "regression" / "quokka-tests.ini"
    if args.suites:
        temp_ini = write_subset_regression_ini(parser, suites, context.worktree_root)
        ini_path = temp_ini

    try:
        with acquire_lock(context, "run", "regression"):
            run_command(
                [str(regtest), "--clean_testdir", str(ini_path)],
                cwd=context.worktree_root,
                command="regression",
                profile=context.profile_name(),
                stdout_to_stderr=context.json_output,
            )
    finally:
        if temp_ini is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_ini.unlink()

    data = {"suites": suites, "profile": context.profile_name(), "ini_file": str(ini_path)}
    text = "Ran {} regression suite(s) in profile {}.".format(len(suites), context.profile_name())
    return CommandResult("regression", context.profile_name(), {"kind": "suite", "name": suites[0] if suites else "*"}, data, text)


def cmd_list(context: CliContext, args: argparse.Namespace) -> CommandResult:
    resource = None
    if args.list_kind == "profiles":
        profile_names = sorted(context.config.profiles)
        data = {"profiles": profile_names}
        text = "\n".join(profile_names)
        return CommandResult("list", None, {"kind": "profiles", "name": "*"}, data, text)

    profile = context.require_profile("list")
    configured = is_build_configured(profile.build_dir)
    if args.list_kind == "problems":
        discovery = "build" if configured else "source"
        problems = discover_problems(profile.build_dir, "list", context.profile_name()) if configured else discover_source_problems(context.worktree_root, profile, "list")
        data = {"problems": problems, "discovery": discovery}
        text = "\n".join(problems)
        resource = {"kind": "problem", "name": "*"}
    elif args.list_kind == "tests":
        discovery = "build" if configured else "source"
        tests = [test.name for test in (discover_tests(profile.build_dir, "list", context.profile_name()) if configured else discover_source_tests(context.worktree_root, profile, "list"))]
        data = {"tests": tests, "discovery": discovery}
        text = "\n".join(tests)
        resource = {"kind": "test", "name": "*"}
    elif args.list_kind == "suites":
        _, suites = discover_suites(context.worktree_root)
        data = {"suites": suites}
        text = "\n".join(suites)
        resource = {"kind": "suite", "name": "*"}
    else:
        raise DiagnosticError("USAGE_ERROR", "Unsupported list kind '{}'.".format(args.list_kind), command="list")
    return CommandResult("list", context.profile_name(), resource, data, text)


def cmd_status(context: CliContext, args: argparse.Namespace) -> CommandResult:
    profile = context.require_profile("status")
    context.resolve_runtime_dir("status")
    context.open_db("status")

    locks = []
    for lock_type in LOCK_TYPES:
        info = inspect_lock(context, lock_type, "status")
        locks.append(
            {
                "lock_type": lock_type,
                "active": info.active,
                "metadata_path": str(info.metadata_path),
                "metadata": info.metadata,
            }
        )

    configured = is_build_configured(profile.build_dir)
    configure_receipt = configure_receipt_path(profile.build_dir)
    define_state = profile_define_state(profile, "status", context.profile_name())
    cache_entries = read_cmake_cache(profile.build_dir, "status", context.profile_name()) if configured else {}
    build_summary = build_summary_from_cache(profile, cache_entries, define_state)
    compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries) if configured else {
        "c": {"path": None, "id": None, "version": None, "metadata_path": None},
        "cxx": {"path": None, "id": None, "version": None, "metadata_path": None},
    }
    configure_state = None
    if configure_receipt.exists():
        configure_data = read_json(configure_receipt, "status", context.profile_name())
        current_fingerprint = compute_configure_fingerprint(context, "status")
        receipt_state = "ready" if configure_data.get("configure_fingerprint") == current_fingerprint else "stale"
        if define_state["mismatches"]:
            receipt_state = "drift"
        receipt_compiler = configure_data.get("compiler")
        if not isinstance(receipt_compiler, dict) or "c" not in receipt_compiler or "cxx" not in receipt_compiler:
            receipt_compiler = compiler_info
        receipt_build = configure_data.get("build")
        if not isinstance(receipt_build, dict) or "build_type" not in receipt_build:
            receipt_build = build_summary
        configure_state = {
            "receipt_path": str(configure_receipt),
            "configured_at": configure_data.get("configured_at"),
            "state": receipt_state,
            "cache_path": define_state["cache_path"],
            "requested_defines": define_state["requested_defines"],
            "effective_defines": define_state["effective_defines"],
            "define_mismatches": define_state["mismatches"],
            "compiler": receipt_compiler,
            "build": receipt_build,
        }
    elif configured:
        configure_state = {
            "receipt_path": str(configure_receipt),
            "configured_at": None,
            "state": "drift" if define_state["mismatches"] else "ready",
            "cache_path": define_state["cache_path"],
            "requested_defines": define_state["requested_defines"],
            "effective_defines": define_state["effective_defines"],
            "define_mismatches": define_state["mismatches"],
            "compiler": compiler_info,
            "build": build_summary,
        }

    problem_states: Dict[str, int] = {"ready": 0, "missing": 0, "stale_source": 0, "stale_configure": 0, "unknown": 0}
    problem_examples: Dict[str, List[Dict[str, Any]]] = {state: [] for state in problem_states}
    repair_hints: Dict[str, str] = {}
    for problem in discover_problems(profile.build_dir, "status", context.profile_name()) if configured else []:
        default_input = default_input_for_problem(context, problem, "status")
        state, details = state_for_artifact(context, problem, "status", default_input)
        problem_states[state] = problem_states.get(state, 0) + 1
        if state == "ready" or len(problem_examples[state]) >= 3:
            continue
        if state == "missing":
            hint = "quokka build {} --profile {}".format(problem, context.profile_name())
            reason = "artifact has not been built for this profile"
        elif state == "stale_source":
            hint = "quokka build {} --profile {}".format(problem, context.profile_name())
            reason = "sources or default input changed since the last build"
        elif state == "stale_configure":
            if details.get("define_mismatches"):
                hint = "fix the profile/CMake drift, then run quokka build {} --profile {} --reconfigure".format(problem, context.profile_name())
                reason = "build configuration drifted from the requested profile"
            else:
                hint = "quokka build {} --profile {} --reconfigure".format(problem, context.profile_name())
                reason = "configure fingerprint changed since the artifact was built"
        else:
            hint = ""
            reason = "artifact metadata is incomplete or unreadable"
        example = {"name": problem, "reason": reason}
        if hint:
            example["repair_hint"] = hint
            repair_hints.setdefault(state, hint)
        problem_examples[state].append(example)

    data = {
        "worktree_root": str(context.worktree_root),
        "worktree_id": context.worktree_id,
        "profile": context.profile_name(),
        "runtime_dir": str(context.resolve_runtime_dir("status")),
        "build_dir": str(profile.build_dir),
        "configured": configured,
        "configure": configure_state,
        "locks": locks,
        "artifacts": problem_states,
        "artifact_examples": problem_examples,
        "repair_hints": repair_hints,
    }
    lines = [
        "worktree: {}".format(data["worktree_root"]),
        "profile: {}".format(context.profile_name()),
        "runtime: {}".format(data["runtime_dir"]),
        "build_dir: {}".format(data["build_dir"]),
        "configured: {}".format("yes" if configured else "no"),
        "locks: {}".format(", ".join("{}={}".format(lock["lock_type"], "active" if lock["active"] else "idle") for lock in locks)),
        "artifacts: ready={ready} missing={missing} stale_source={stale_source} stale_configure={stale_configure} unknown={unknown}".format(**problem_states),
    ]
    if configure_state is not None and configure_state.get("define_mismatches"):
        lines.append(
            "configure: drift ({})".format(format_define_mismatch_summary(configure_state["define_mismatches"]))
        )
    for state in ("missing", "stale_source", "stale_configure", "unknown"):
        examples = problem_examples.get(state) or []
        if not examples:
            continue
        hidden = problem_states.get(state, 0) - len(examples)
        suffix = " (+{} more)".format(hidden) if hidden > 0 else ""
        lines.append("{} examples: {}{}".format(state, ", ".join(example["name"] for example in examples), suffix))
        hint = repair_hints.get(state)
        if hint:
            lines.append("{} repair: {}".format(state, hint))
    return CommandResult("status", context.profile_name(), None, data, "\n".join(lines))


def cmd_tidy(context: CliContext, args: argparse.Namespace) -> CommandResult:
    profile = context.require_profile("tidy")
    selector = args.selector or "changed"
    if selector not in TIDY_SELECTORS:
        raise DiagnosticError(
            "TIDY_SELECTOR_INVALID",
            "Unsupported tidy selector '{}'.".format(selector),
            command="tidy",
            profile=context.profile_name(),
        )
    context.resolve_runtime_dir("tidy")
    ensure_no_conflicting_locks(context, ("build",), "tidy")

    compile_commands = profile.build_dir / "compile_commands.json"
    if not compile_commands.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' does not have compile_commands.json yet.".format(context.profile_name()),
            command="tidy",
            profile=context.profile_name(),
            details={"compile_commands": str(compile_commands)},
        )

    script = context.worktree_root / "scripts" / "bash" / "tidy.sh"
    cmd = [str(script)]
    if args.fix:
        cmd.append("--fix")
    cmd.extend([str(profile.build_dir), selector])
    run_command(cmd, cwd=context.worktree_root, command="tidy", profile=context.profile_name(), stdout_to_stderr=context.json_output)

    data = {"build_dir": str(profile.build_dir), "selector": selector, "fix": bool(args.fix)}
    text = "Ran clang-tidy wrapper for profile {} with selector '{}'.".format(context.profile_name(), selector)
    return CommandResult("tidy", context.profile_name(), None, data, text)


def cmd_format(context: CliContext, args: argparse.Namespace) -> CommandResult:
    selector = args.selector or "changed"
    if selector not in FORMAT_SELECTORS:
        raise DiagnosticError(
            "FORMAT_SELECTOR_INVALID",
            "Unsupported format selector '{}'.".format(selector),
            command="format",
        )
    context.resolve_runtime_dir("format")
    ensure_no_conflicting_locks(context, ("build",), "format")

    if shutil.which("pre-commit") is None:
        raise DiagnosticError(
            "PRE_COMMIT_UNAVAILABLE",
            "pre-commit is required but not installed.",
            command="format",
        )

    if selector == "all":
        run_command(
            ["pre-commit", "run", "clang-format", "--all-files"],
            cwd=context.worktree_root,
            command="format",
            profile=None,
            stdout_to_stderr=context.json_output,
        )
        data = {"selector": selector, "files": [], "all_files": True}
        text = "Ran clang-format hook over all eligible files."
        return CommandResult("format", None, None, data, text)

    files = git_changed_files(context.worktree_root, selector, "format", None)
    if not files:
        data = {"selector": selector, "files": [], "no_op": True}
        return CommandResult("format", None, None, data, "No files selected for formatting.")

    run_command(
        ["pre-commit", "run", "clang-format", "--files"] + files,
        cwd=context.worktree_root,
        command="format",
        profile=None,
        stdout_to_stderr=context.json_output,
    )
    data = {"selector": selector, "files": files}
    text = "Ran clang-format hook on {} file(s).".format(len(files))
    return CommandResult("format", None, None, data, text)


def cmd_lock(context: CliContext, args: argparse.Namespace) -> CommandResult:
    context.resolve_runtime_dir("lock")
    context.open_db("lock")
    if args.lock_action == "ls":
        locks = []
        for lock_type in LOCK_TYPES:
            info = inspect_lock(context, lock_type, "lock")
            locks.append(
                {
                    "lock_type": lock_type,
                    "active": info.active,
                    "lock_path": str(info.lock_path),
                    "metadata_path": str(info.metadata_path),
                    "metadata": info.metadata,
                }
            )
        text = "\n".join(
            "{}: {}".format(lock["lock_type"], "active" if lock["active"] else "idle") for lock in locks
        )
        return CommandResult("lock", context.profile_name(), None, {"locks": locks}, text)

    broken = break_locks(context, "lock")
    text = "Removed {} stale lock(s).".format(len(broken))
    return CommandResult("lock", context.profile_name(), None, {"broken": broken}, text)


def cmd_clean(context: CliContext, args: argparse.Namespace) -> CommandResult:
    action = args.clean_kind
    if action == "runs":
        runs_dir = context.resolve_runtime_dir("clean") / "runs" / "wt-{}".format(context.worktree_id)
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
        data = {"removed": str(runs_dir), "existed": runs_dir.exists()}
        return CommandResult("clean", context.profile_name(), None, data, "Cleaned run scratch state.")

    if action == "locks":
        broken = break_locks(context, "clean")
        return CommandResult("clean", context.profile_name(), None, {"broken": broken}, "Removed {} stale lock(s).".format(len(broken)))

    profile = context.require_profile("clean")
    removed: List[str] = []
    for path in sorted(artifact_receipts_dir(profile.build_dir).glob("*.json")):
        path.unlink()
        removed.append(str(path))
    configure_path = configure_receipt_path(profile.build_dir)
    if configure_path.exists():
        configure_path.unlink()
        removed.append(str(configure_path))
    data = {"removed": removed}
    text = "Removed {} receipt file(s) for profile {}.".format(len(removed), context.profile_name())
    return CommandResult("clean", context.profile_name(), None, data, text)


def cmd_doctor(context: CliContext, args: argparse.Namespace) -> CommandResult:
    topic = args.topic or "all"
    data: Dict[str, Any] = {}
    lines: List[str] = []
    profile: Optional[ProfileConfig] = None
    configured = False
    cache_entries: Dict[str, Dict[str, str]] = {}
    define_state: Optional[Dict[str, Any]] = None
    python_probe: Optional[Dict[str, Any]] = None

    if topic in ("all", "runtime", "profile"):
        profile = context.require_profile("doctor")
        configured = is_build_configured(profile.build_dir)
        define_state = profile_define_state(profile, "doctor", context.profile_name())
        if configured:
            cache_entries = read_cmake_cache(profile.build_dir, "doctor", context.profile_name())
        python_executable, python_source = doctor_python_executable(cache_entries)
        python_probe = probe_python_stack(python_executable, python_source)

    if topic in ("all", "runtime"):
        runtime_dir = context.resolve_runtime_dir("doctor")
        db = context.open_db("doctor")
        assert profile is not None
        assert python_probe is not None
        tools = {
            "cmake": tool_probe("cmake", ["--version"]),
            "ctest": tool_probe("ctest", ["--version"]),
            "git": tool_probe("git", ["--version"]),
            "generator": generator_tool_probe(profile.generator),
        }
        data["runtime"] = {
            "runtime_dir": str(runtime_dir),
            "state_db": str(context.db_path("doctor")),
            "sqlite_ok": db.execute("SELECT 1").fetchone()[0] == 1,
            "tools": tools,
            "python": python_probe,
        }
        lines.append("runtime: ok ({})".format(runtime_dir))
        lines.append(
            "tools: cmake={cmake}, ctest={ctest}, git={git}, {generator_label}={generator_status}".format(
                cmake=tools["cmake"]["status"],
                ctest=tools["ctest"]["status"],
                git=tools["git"]["status"],
                generator_label=profile.generator,
                generator_status=tools["generator"]["status"],
            )
        )
        plotting_state = "ok" if python_probe["plotting_available"] else "unavailable"
        plotting_detail = ""
        if python_probe["failed_modules"]:
            plotting_detail = " ({})".format(", ".join(python_probe["failed_modules"]))
        lines.append(
            "python: interpreter={} numpy={} plotting={}{}".format(
                python_probe["executable"] or "<missing>",
                "ok" if python_probe["numpy_available"] else "missing",
                plotting_state,
                plotting_detail,
            )
        )

    if topic in ("all", "locking"):
        locks = []
        for lock_type in LOCK_TYPES:
            info = inspect_lock(context, lock_type, "doctor")
            locks.append({"lock_type": lock_type, "active": info.active, "metadata": info.metadata})
        data["locking"] = {"locks": locks}
        lines.append("locking: {}".format(", ".join("{}={}".format(lock["lock_type"], "active" if lock["active"] else "idle") for lock in locks)))

    if topic in ("all", "profile"):
        assert profile is not None
        assert define_state is not None
        assert python_probe is not None
        compile_commands = profile.build_dir / "compile_commands.json"
        build_summary = build_summary_from_cache(profile, cache_entries, define_state)
        build_summary["python_executable"] = build_summary["python_executable"] or python_probe["executable"]
        build_summary["c_compiler"] = cache_entry_value(cache_entries, "CMAKE_C_COMPILER")
        build_summary["cxx_compiler"] = cache_entry_value(cache_entries, "CMAKE_CXX_COMPILER")
        compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries) if configured else {
            "c": {"path": None, "id": None, "version": None, "metadata_path": None},
            "cxx": {"path": None, "id": None, "version": None, "metadata_path": None},
        }
        data["profile"] = {
            "profile": context.profile_name(),
            "build_dir": str(profile.build_dir),
            "configured": configured,
            "compile_commands": str(compile_commands),
            "compile_commands_exists": compile_commands.exists(),
            "cache_path": define_state["cache_path"],
            "requested_defines": define_state["requested_defines"],
            "effective_defines": define_state["effective_defines"],
            "define_mismatches": define_state["mismatches"],
            "build": build_summary,
            "compiler": compiler_info,
            "python": python_probe,
        }
        if define_state["mismatches"]:
            lines.append("profile: {} (configured, drift)".format(context.profile_name()))
            lines.append("profile drift: {}".format(format_define_mismatch_summary(define_state["mismatches"])))
        else:
            lines.append("profile: {} ({})".format(context.profile_name(), "configured" if configured else "unconfigured"))
        lines.append(
            "profile build: type={} mpi={} gpu={} cxx={}".format(
                build_summary["build_type"] or "<unknown>",
                build_summary["mpi"] or "<unknown>",
                build_summary["gpu_backend"] or "<unknown>",
                build_summary["cxx_compiler"] or "<unknown>",
            )
        )
        lines.append(
            "compile_commands: {} ({})".format(
                "present" if compile_commands.exists() else "missing",
                compile_commands,
            )
        )
        if build_summary["hdf5_dir"] or build_summary["python_executable"]:
            lines.append(
                "profile deps: hdf5={} python={}".format(
                    build_summary["hdf5_dir"] or "<unknown>",
                    build_summary["python_executable"] or "<unknown>",
                )
            )

    return CommandResult("doctor", context.profile_name(), None, data, "\n".join(lines))


def cmd_activation_env(context: CliContext, args: argparse.Namespace) -> CommandResult:
    profile = context.require_profile("_activate-env")
    runtime_dir = context.resolve_runtime_dir("_activate-env")
    prompt = "(quokka:{}@{})".format(context.worktree_root.name, profile.name)
    exports = {
        "QUOKKA_ACTIVE": "1",
        "QUOKKA_WORKTREE_ROOT": str(context.worktree_root),
        "QUOKKA_WORKTREE_ID": context.worktree_id,
        "QUOKKA_PROFILE": profile.name,
        "QUOKKA_RUNTIME_DIR": str(runtime_dir),
        "QUOKKA_PROMPT_PREFIX": prompt,
    }
    lines = ["export {}={}".format(key, shlex.quote(value)) for key, value in exports.items()]
    return CommandResult("_activate-env", profile.name, None, {"exports": exports}, "\n".join(lines))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quokka")
    parser.add_argument("-C", "--worktree", help="Path to the target worktree.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", help="Profile name from quokka.toml.")
    common.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    common_no_profile = argparse.ArgumentParser(add_help=False)
    common_no_profile.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", parents=[common])
    build.add_argument("targets", nargs="*")
    build.add_argument("--reconfigure", action="store_true")
    build.set_defaults(handler=cmd_build)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("problem")
    run.add_argument("--input")
    run.add_argument("--build-if-needed", action="store_true")
    run.set_defaults(handler=cmd_run)

    test = subparsers.add_parser("test", parents=[common])
    test.add_argument("test_name", nargs="?")
    test.add_argument("--ctest-regex")
    test.add_argument("--build-if-needed", action="store_true")
    test.add_argument("--stream", action="store_true", help="Stream live test progress and stdout/stderr.")
    test.set_defaults(handler=cmd_test)

    regression = subparsers.add_parser("regression", parents=[common])
    regression.add_argument("suites", nargs="*")
    regression.add_argument("--build-if-needed", action="store_true")
    regression.set_defaults(handler=cmd_regression)

    tidy = subparsers.add_parser("tidy", parents=[common])
    tidy.add_argument("selector", nargs="?")
    tidy.add_argument("--fix", action="store_true")
    tidy.set_defaults(handler=cmd_tidy)

    fmt = subparsers.add_parser("format", parents=[common_no_profile])
    fmt.add_argument("selector", nargs="?")
    fmt.set_defaults(handler=cmd_format)

    list_cmd = subparsers.add_parser("list", parents=[common])
    list_cmd.add_argument("list_kind", choices=["problems", "tests", "suites", "profiles"])
    list_cmd.set_defaults(handler=cmd_list)

    status = subparsers.add_parser("status", parents=[common])
    status.set_defaults(handler=cmd_status)

    lock = subparsers.add_parser("lock", parents=[common])
    lock.add_argument("lock_action", choices=["ls", "break"])
    lock.add_argument("--scope")
    lock.set_defaults(handler=cmd_lock)

    clean = subparsers.add_parser("clean", parents=[common])
    clean.add_argument("clean_kind", choices=["runs", "locks", "profile"])
    clean.set_defaults(handler=cmd_clean)

    doctor = subparsers.add_parser("doctor", parents=[common])
    doctor.add_argument("topic", nargs="?", choices=["locking", "runtime", "profile"])
    doctor.set_defaults(handler=cmd_doctor)

    activation = subparsers.add_parser("_activate-env", parents=[common], help=argparse.SUPPRESS)
    activation.set_defaults(handler=cmd_activation_env)

    return parser


def context_for_args(args: argparse.Namespace) -> CliContext:
    worktree_root = resolve_worktree_root(args)
    config = load_repo_config(worktree_root)
    profile_name = getattr(args, "profile", None)
    json_output = bool(getattr(args, "json", False))
    profile = None
    if args.command in {"format"}:
        profile = None
    elif args.command == "list" and args.list_kind == "profiles":
        profile = None
    else:
        profile = resolve_profile(config, profile_name, args.command)
    return CliContext(worktree_root, config, profile, json_output=json_output)


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    command_name = args.command
    json_output = bool(getattr(args, "json", False))
    profile_name = getattr(args, "profile", None)

    try:
        context = context_for_args(args)
        result = args.handler(context, args)
        print(format_result(result, json_output))
        return 0
    except DiagnosticError as exc:
        output = error_result(exc, json_output, command_name, profile_name)
        stream = sys.stdout if json_output else sys.stderr
        print(output, file=stream)
        return exc.exit_code
    except Exception as exc:
        if os.environ.get("QUOKKA_DEBUG"):
            traceback.print_exc()
        error = DiagnosticError(
            "INTERNAL_ERROR",
            "Unexpected CLI failure: {}".format(exc),
            command=command_name,
            profile=profile_name,
        )
        output = error_result(error, json_output, command_name, profile_name)
        stream = sys.stdout if json_output else sys.stderr
        print(output, file=stream)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
