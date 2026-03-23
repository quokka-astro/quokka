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

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from quokka.core.constants import IGNORED_PROFILE_DEFINES

from quokka.core.errors import DiagnosticError

from quokka.core.types import ProfileConfig, RepoConfig

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
