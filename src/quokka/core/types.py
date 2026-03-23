from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class CommandRequest:
    worktree: str | None = None
    profile: str | None = None
    json_output: bool = False

    command_name: ClassVar[str] = ""

    def resolves_profile(self) -> bool:
        return True


@dataclass(frozen=True)
class ActivationRequest(CommandRequest):
    command_name: ClassVar[str] = "_activate-env"


@dataclass(frozen=True)
class BootstrapRequest(CommandRequest):
    fix: bool = False
    include_optional: bool = False

    command_name: ClassVar[str] = "bootstrap"


@dataclass(frozen=True)
class BuildRequest(CommandRequest):
    targets: tuple[str, ...] = ()
    reconfigure: bool = False

    command_name: ClassVar[str] = "build"


@dataclass(frozen=True)
class CleanRequest(CommandRequest):
    clean_kind: str = "runs"

    command_name: ClassVar[str] = "clean"


@dataclass(frozen=True)
class ConfigureRequest(CommandRequest):
    reconfigure: bool = False

    command_name: ClassVar[str] = "configure"


@dataclass(frozen=True)
class DoctorRequest(CommandRequest):
    topic: str = "profile"

    command_name: ClassVar[str] = "doctor"


@dataclass(frozen=True)
class FormatRequest(CommandRequest):
    selector: str | None = None

    command_name: ClassVar[str] = "format"

    def resolves_profile(self) -> bool:
        return False


@dataclass(frozen=True)
class ListRequest(CommandRequest):
    list_kind: str = "profiles"

    command_name: ClassVar[str] = "list"

    def resolves_profile(self) -> bool:
        return self.list_kind != "profiles"


@dataclass(frozen=True)
class LockRequest(CommandRequest):
    lock_action: str = "ls"
    scope: str | None = None

    command_name: ClassVar[str] = "lock"


@dataclass(frozen=True)
class RunRequest(CommandRequest):
    problem: str = ""
    input: str | None = None
    build_if_needed: bool = False
    verbose_runtime: bool = False

    command_name: ClassVar[str] = "run"


@dataclass(frozen=True)
class SmokeRequest(CommandRequest):
    test_name: str | None = None
    stream: bool = False
    compact_stream: bool = False

    command_name: ClassVar[str] = "smoke"


@dataclass(frozen=True)
class StatusRequest(CommandRequest):
    command_name: ClassVar[str] = "status"


@dataclass(frozen=True)
class TestRequest(CommandRequest):
    test_name: str | None = None
    ctest_regex: str | None = None
    build_if_needed: bool = False
    stream: bool = False
    compact_stream: bool = False

    command_name: ClassVar[str] = "test"


@dataclass(frozen=True)
class TidyRequest(CommandRequest):
    selector: str | None = None
    fix: bool = False

    command_name: ClassVar[str] = "tidy"


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    build_dir: Path
    generator: str
    defines: dict[str, str]
    executor_kind: str
    executor: dict[str, Any]


@dataclass(frozen=True)
class RepoConfig:
    path: Path
    default_profile: str
    profiles: dict[str, ProfileConfig]
    policy: dict[str, Any]


@dataclass(frozen=True)
class TestSpec:
    name: str
    command: list[str]
    working_directory: Path | None
    source_path: Path
