from __future__ import annotations

import argparse
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
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from quokka.core.constants import LOCK_TYPES, SCHEMA

from quokka.core.errors import DiagnosticError

from quokka.core.subprocess import command_output

from quokka.core.types import ProfileConfig, TestSpec

from quokka.project.root import utc_now

from quokka.tools.cmake import (
    build_summary_from_cache,
    cmake_cache_path,
    cmake_version,
    compiler_metadata_from_build,
    compute_configure_fingerprint,
    profile_define_state,
    read_cmake_cache,
)

from quokka.vcs.git import compute_source_fingerprint, git_metadata

from quokka.model.files import default_input_for_problem, relative_or_absolute

from quokka.model.tests import discover_tests, test_map_by_name

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
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)

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

def read_optional_json(path: Path, command: str, profile: Optional[str]) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "State file '{}' is unreadable.".format(path),
            command=command,
            profile=profile,
            details={"path": str(path)},
        ) from exc

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

    define_state = profile_define_state(profile, command, context.profile_name())
    cache_entries = read_cmake_cache(profile.build_dir, command, context.profile_name()) if cmake_cache_path(profile.build_dir).exists() else {}
    build_summary = build_summary_from_cache(profile, cache_entries, define_state, context.worktree_root)
    compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries) if cache_entries else {
        "c": {"path": None, "source": None, "id": None, "version": None, "metadata_path": None},
        "cxx": {"path": None, "source": None, "id": None, "version": None, "metadata_path": None},
    }
    configure_data = read_optional_json(configure_receipt_path(profile.build_dir), command, context.profile_name())
    if configure_data is None:
        return "stale_configure", {
            "receipt_path": str(receipt_path),
            "configure_receipt_path": str(configure_receipt_path(profile.build_dir)),
            "reason": "configure receipt is missing",
        }
    if not configure_receipt_matches_current_build(configure_data, build_summary, compiler_info):
        return "stale_configure", {
            "receipt_path": str(receipt_path),
            "configure_receipt_path": str(configure_receipt_path(profile.build_dir)),
            "reason": "configure receipt does not match the current build cache",
        }
    accepted_fingerprints = set(configure_receipt_fingerprint_aliases(configure_data))
    if receipt_configure not in accepted_fingerprints:
        return "stale_configure", {
            "receipt_path": str(receipt_path),
            "configure_fingerprint_previous": receipt_configure,
            "configure_fingerprint_current": configure_data.get("configure_fingerprint"),
            "configure_fingerprint_aliases": sorted(accepted_fingerprints),
        }

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
    build_summary = build_summary_from_cache(profile, cache_entries, define_state, context.worktree_root)
    compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries)
    previous_receipt = read_optional_json(configure_receipt_path(profile.build_dir), command, context.profile_name())
    current_fingerprint = compute_configure_fingerprint(context, command)
    fingerprint_aliases: List[str] = []
    if previous_receipt is not None and configure_receipt_matches_current_build(previous_receipt, build_summary, compiler_info):
        for fingerprint in configure_receipt_fingerprint_aliases(previous_receipt):
            if fingerprint != current_fingerprint and fingerprint not in fingerprint_aliases:
                fingerprint_aliases.append(fingerprint)
    payload = {
        "schema": 1,
        "configured_at": utc_now(),
        "configure_fingerprint": current_fingerprint,
        "configure_fingerprint_aliases": fingerprint_aliases,
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

def receipt_subset_matches(receipt_data: Dict[str, Any], current_data: Dict[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        if key not in receipt_data or receipt_data.get(key) is None:
            continue
        if receipt_data.get(key) != current_data.get(key):
            return False
    return True

def configure_receipt_matches_current_build(
    configure_data: Dict[str, Any],
    build_summary: Dict[str, Any],
    compiler_info: Dict[str, Dict[str, Optional[str]]],
) -> bool:
    receipt_build = configure_data.get("build")
    receipt_compiler = configure_data.get("compiler")
    if not isinstance(receipt_build, dict) or not isinstance(receipt_compiler, dict):
        return False
    if not receipt_subset_matches(
        receipt_build,
        build_summary,
        ("generator", "build_type", "mpi", "space_dim", "gpu_backend", "hdf5_dir", "hdf5_diff", "python_enabled", "python_executable"),
    ):
        return False
    for key in ("c", "cxx"):
        receipt_language = receipt_compiler.get(key)
        current_language = compiler_info.get(key)
        if not isinstance(receipt_language, dict) or not isinstance(current_language, dict):
            return False
        if not receipt_subset_matches(receipt_language, current_language, ("path", "id", "version")):
            return False
    return True

def configure_receipt_fingerprint_aliases(configure_data: Dict[str, Any]) -> List[str]:
    fingerprints: List[str] = []
    primary = configure_data.get("configure_fingerprint")
    if isinstance(primary, str) and primary:
        fingerprints.append(primary)
    aliases = configure_data.get("configure_fingerprint_aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias and alias not in fingerprints:
                fingerprints.append(alias)
    return fingerprints

def build_lock_paths(context: CliContext, lock_type: str, command: str, *, create_runtime: bool = True) -> Tuple[Path, Path]:
    runtime_dir = context.resolve_runtime_dir(command, create=create_runtime)
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

def lock_metadata_is_active(context: CliContext, metadata: Optional[Dict[str, Any]]) -> bool:
    if not metadata:
        return False
    pid = metadata.get("pid")
    if not isinstance(pid, int):
        return False
    hostname = metadata.get("hostname")
    if isinstance(hostname, str) and hostname and hostname != context.hostname:
        return False
    boot_id = metadata.get("boot_id")
    current = current_boot_id()
    if isinstance(boot_id, str) and boot_id and current and boot_id != current:
        return False
    return pid_is_alive(pid)

def inspect_lock(context: CliContext, lock_type: str, command: str, *, probe_active: bool = True) -> LockInfo:
    lock_path, metadata_path = build_lock_paths(context, lock_type, command, create_runtime=probe_active)
    metadata = read_optional_json(metadata_path, command, context.profile_name())
    if not probe_active:
        active = lock_path.exists() and lock_metadata_is_active(context, metadata)
        return LockInfo(lock_type=lock_type, lock_path=lock_path, metadata_path=metadata_path, active=active, metadata=metadata)

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
    context.resolve_runtime_dir(command)
    context.open_db(command)
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
    return buildtree_state(build_dir)["configured"]
