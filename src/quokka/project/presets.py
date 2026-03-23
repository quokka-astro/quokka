from __future__ import annotations

import os

from quokka.core.errors import DiagnosticError
from quokka.core.types import ProfileConfig, RepoConfig


def resolve_profile(config: RepoConfig, profile_name: str | None, command: str) -> ProfileConfig:
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


def list_profiles(config: RepoConfig) -> list[str]:
    return sorted(config.profiles)
