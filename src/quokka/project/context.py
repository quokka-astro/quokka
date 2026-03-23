from __future__ import annotations

import hashlib
import os
import socket
import sqlite3
from pathlib import Path

from quokka.core.constants import SCHEMA
from quokka.core.errors import DiagnosticError
from quokka.core.types import CommandRequest, ProfileConfig, RepoConfig
from quokka.project.config import load_repo_config
from quokka.project.presets import resolve_profile
from quokka.project.root import canonical_path, current_uid_or_user, ensure_runtime_dir_layout, is_subpath, resolve_worktree_root, utc_now


class ProjectContext:
    def __init__(self, worktree_root: Path, config: RepoConfig, profile: ProfileConfig | None, *, json_output: bool = False) -> None:
        self.worktree_root = worktree_root
        self.config = config
        self.profile = profile
        self.json_output = json_output
        self.hostname = socket.gethostname()
        self.worktree_id = hashlib.sha256((self.hostname + "\n" + str(self.worktree_root)).encode("utf-8")).hexdigest()[:12]
        self._runtime_dir: Path | None = None
        self._runtime_dir_ready = False
        self._db: sqlite3.Connection | None = None

    def profile_name(self) -> str | None:
        return None if self.profile is None else self.profile.name

    def require_profile(self, command: str) -> ProfileConfig:
        if self.profile is None:
            raise DiagnosticError(
                "UNKNOWN_PROFILE",
                "No profile is selected for this command.",
                command=command,
            )
        return self.profile

    def resolve_runtime_dir(self, command: str, *, create: bool = True) -> Path:
        if self._runtime_dir is not None:
            if create and not self._runtime_dir_ready:
                try:
                    ensure_runtime_dir_layout(self._runtime_dir)
                except OSError as exc:
                    raise DiagnosticError(
                        "TOOL_FAILED",
                        "Unable to create a usable runtime directory.",
                        command=command,
                        profile=self.profile_name(),
                        details={
                            "attempts": [
                                {
                                    "runtime_dir": str(self._runtime_dir),
                                    "resolved_runtime_dir": str(canonical_path(self._runtime_dir)),
                                    "error": str(exc),
                                }
                            ]
                        },
                    ) from exc
                self._runtime_dir_ready = True
            return self._runtime_dir

        env_runtime = os.environ.get("QUOKKA_RUNTIME_DIR")
        runtime_candidates: list[Path] = []
        if env_runtime:
            runtime_candidates.append(Path(env_runtime).expanduser())
        else:
            xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
            if xdg_runtime:
                runtime_candidates.append(Path(xdg_runtime).expanduser() / "quokka")
            runtime_candidates.append(Path("/tmp") / ("quokka-" + current_uid_or_user()))

        worktree_root = canonical_path(self.worktree_root)
        home_dir = canonical_path(Path.home())
        errors: list[dict[str, str]] = []
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

            if not create:
                self._runtime_dir = runtime_dir
                self._runtime_dir_ready = runtime_dir.exists()
                return runtime_dir

            try:
                ensure_runtime_dir_layout(runtime_dir)
                self._runtime_dir = runtime_dir
                self._runtime_dir_ready = True
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

    def db_path(self, command: str, *, create: bool = True) -> Path:
        return self.resolve_runtime_dir(command, create=create) / "state.db"

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

    def update_profile_index(self, configure_fingerprint: str | None, command: str) -> None:
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

    def update_artifact_index(self, artifact_id: str, receipt: dict[str, object], command: str) -> None:
        profile = self.require_profile(command)
        receipt_path = profile.build_dir / ".quokka" / "artifacts" / "{}.json".format(artifact_id)
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
                str(receipt_path),
                receipt["source_fingerprint"],
                receipt["configure_fingerprint"],
                receipt["built_at"],
            ),
        )
        db.commit()

    def update_lock_index(self, lock_type: str, metadata: dict[str, object], command: str) -> None:
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


def context_for_request(request: CommandRequest) -> ProjectContext:
    worktree_root = resolve_worktree_root(request.worktree)
    config = load_repo_config(worktree_root)
    profile = resolve_profile(config, request.profile, request.command_name) if request.resolves_profile() else None
    return ProjectContext(worktree_root, config, profile, json_output=request.json_output)


CliContext = ProjectContext
