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

from quokka.core.constants import ENV_OVERRIDE_KEYS, PYTHON_MODULE_PACKAGES

from quokka.core.errors import DiagnosticError

from quokka.core.subprocess import command_output, first_nonempty_line, resolve_executable_path, run_probe_command, shell_join

from quokka.core.types import ProfileConfig

from quokka.project.config import normalize_define_value

from quokka.project.root import is_subpath

from quokka.tools.ctest import ctest_lasttest_log_path

from quokka.project.root import utc_now

def python_install_hint(python_probe: Dict[str, Any]) -> Optional[str]:
    failed_modules = python_probe.get("failed_modules") or []
    packages: List[str] = []
    for module_name in failed_modules:
        package = PYTHON_MODULE_PACKAGES.get(str(module_name))
        if package and package not in packages:
            packages.append(package)
    if not packages:
        return None
    executable = python_probe.get("executable") or "python3"
    return "{} -m pip install {}".format(shlex.quote(str(executable)), " ".join(packages))

def preferred_pre_commit_install_command() -> Optional[List[str]]:
    in_virtualenv = bool(os.environ.get("VIRTUAL_ENV"))
    uv_path = resolve_executable_path("uv")
    python_path = resolve_executable_path("python3") or resolve_executable_path("python") or sys.executable
    if uv_path is not None:
        command = [uv_path, "pip", "install"]
        if not in_virtualenv:
            command.append("--user")
        command.append("pre-commit")
        return command
    if python_path:
        command = [python_path, "-m", "pip", "install"]
        if not in_virtualenv:
            command.append("--user")
        command.append("pre-commit")
        return command
    return None

def pre_commit_install_commands() -> List[str]:
    commands: List[str] = []
    in_virtualenv = bool(os.environ.get("VIRTUAL_ENV"))
    uv_path = resolve_executable_path("uv")
    python_path = resolve_executable_path("python3") or resolve_executable_path("python") or sys.executable
    if uv_path is not None:
        uv_command = [uv_path, "pip", "install"]
        if not in_virtualenv:
            uv_command.append("--user")
        uv_command.append("pre-commit")
        commands.append(shell_join(uv_command))
    if python_path:
        python_command = [python_path, "-m", "pip", "install"]
        if not in_virtualenv:
            python_command.append("--user")
        python_command.append("pre-commit")
        rendered = shell_join(python_command)
        if rendered not in commands:
            commands.append(rendered)
    return commands

def cmake_bool_state(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    upper = str(value).strip().upper()
    if upper in {"1", "ON", "TRUE", "YES", "Y"}:
        return True
    if upper in {"0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND", ""}:
        return False
    return None

def repo_default_amrex_mpi(worktree_root: Path) -> Optional[str]:
    cmake_path = worktree_root / "CMakeLists.txt"
    try:
        lines = cmake_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    pattern = re.compile(r"set\s*\(\s*AMReX_MPI\s+([^\s\)]+)", re.IGNORECASE)
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match is None:
            continue
        return normalize_define_value(match.group(1).strip("\"'"))
    return None

def resolve_mpi_requirement(
    worktree_root: Path,
    define_state: Optional[Dict[str, Any]],
    cache_entries: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    cached = cache_entry_value(cache_entries, "AMReX_MPI")
    if cached is not None:
        return {"value": cached, "enabled": cmake_bool_state(cached), "source": "CMake cache"}

    requested = None
    if define_state is not None:
        requested = (define_state.get("requested_defines") or {}).get("AMReX_MPI")
    if requested is not None:
        return {"value": requested, "enabled": cmake_bool_state(requested), "source": "profile define"}

    repo_default = repo_default_amrex_mpi(worktree_root)
    if repo_default is not None:
        return {"value": repo_default, "enabled": cmake_bool_state(repo_default), "source": "repo CMake default"}

    return {"value": None, "enabled": None, "source": "unknown"}

def mpi_install_hint(mpi_state: Dict[str, Any]) -> Optional[str]:
    if mpi_state.get("enabled") is False:
        return None

    missing_required = list(mpi_state.get("missing_required") or [])
    if missing_required:
        verb = "is" if len(missing_required) == 1 else "are"
        tools = ", ".join(missing_required)
        return "install or load an MPI toolchain so {} {} on PATH".format(tools, verb)

    missing_optional = list(mpi_state.get("missing_optional") or [])
    if missing_optional:
        verb = "is" if len(missing_optional) == 1 else "are"
        tools = ", ".join(missing_optional)
        return "ensure {} {} on PATH for local multi-rank runs and tests".format(tools, verb)

    return None

def collect_mpi_state(
    context: "CliContext",
    command: str,
    *,
    cache_entries: Optional[Dict[str, Dict[str, str]]] = None,
    define_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = context.require_profile(command)
    resolved_cache_entries = cache_entries
    if resolved_cache_entries is None:
        resolved_cache_entries = {}
        if cmake_cache_path(profile.build_dir).exists():
            resolved_cache_entries = read_cmake_cache(profile.build_dir, command, context.profile_name())
    if define_state is None:
        define_state = profile_define_state(profile, command, context.profile_name())

    requirement = resolve_mpi_requirement(context.worktree_root, define_state, resolved_cache_entries)
    c_wrapper = tool_probe("mpicc", ["--version"], label="mpicc")
    cxx_wrapper = tool_probe("mpicxx", ["--version"], label="mpicxx")
    launcher = tool_probe("mpirun", ["--version"], label="mpirun")

    missing_required: List[str] = []
    missing_optional: List[str] = []
    enabled = requirement["enabled"]
    if enabled is not False:
        for probe in (c_wrapper, cxx_wrapper):
            if probe["status"] != "ok":
                missing_required.append(str(probe["tool"]))
        if launcher["status"] != "ok":
            missing_optional.append(str(launcher["tool"]))

    if enabled is False:
        status = "disabled"
    elif missing_required:
        status = "missing"
    elif missing_optional:
        status = "partial"
    elif enabled is None:
        status = "unknown"
    else:
        status = "ok"

    result = {
        "setting": requirement["value"],
        "enabled": enabled,
        "source": requirement["source"],
        "status": status,
        "wrappers": {"c": c_wrapper, "cxx": cxx_wrapper},
        "launcher": launcher,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }
    result["install_hint"] = mpi_install_hint(result)
    return result

def cmake_version(command: str, profile: Optional[str]) -> str:
    output = command_output(["cmake", "--version"], command=command, profile=profile)
    first_line = output.splitlines()[0] if output else ""
    parts = first_line.split()
    return parts[2] if len(parts) >= 3 else first_line

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

def ctest_root_testfile_path(build_dir: Path) -> Path:
    return build_dir / "CTestTestfile.cmake"

def buildtree_state(build_dir: Path) -> Dict[str, bool]:
    cmake_cache_exists = cmake_cache_path(build_dir).exists()
    ctest_metadata_exists = ctest_root_testfile_path(build_dir).exists()
    return {
        "cmake_cache_exists": cmake_cache_exists,
        "ctest_metadata_exists": ctest_metadata_exists,
        "configured": cmake_cache_exists and ctest_metadata_exists,
        "partial_configure": cmake_cache_exists and not ctest_metadata_exists,
    }

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
            "source": "cache" if cache_entry_value(cache_entries, "CMAKE_{}_COMPILER".format(language)) else None,
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

def active_env_overrides() -> Dict[str, str]:
    return {key: os.environ.get(key) for key in ENV_OVERRIDE_KEYS if os.environ.get(key)}

def toolchain_env_overrides_for_text(env_overrides: Dict[str, str]) -> Dict[str, str]:
    keys = ("CC", "CXX", "CMAKE_GENERATOR", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER")
    return {key: env_overrides[key] for key in keys if key in env_overrides}

def populate_compiler_candidates_from_env(
    compiler_info: Dict[str, Dict[str, Optional[str]]],
    env_overrides: Dict[str, str],
) -> Dict[str, Dict[str, Optional[str]]]:
    for language, key, env_keys in (
        ("C", "c", ("CMAKE_C_COMPILER", "CC")),
        ("CXX", "cxx", ("CMAKE_CXX_COMPILER", "CXX")),
    ):
        info = compiler_info[key]
        if info.get("path"):
            continue
        for env_key in env_keys:
            value = env_overrides.get(env_key)
            if value:
                info["path"] = value
                info["source"] = "env:{}".format(env_key)
                break
    return compiler_info

def format_tool_probe_short(tool: Dict[str, Any]) -> str:
    path = tool.get("path")
    version = tool.get("version")
    if path and version:
        return "{} ({})".format(path, version)
    if path:
        return str(path)
    detail = tool.get("detail")
    if detail:
        return "<{}>".format(detail)
    return "<missing>"

def format_compiler_toolchain(info: Dict[str, Optional[str]], *, configured: bool) -> str:
    path = info.get("path")
    compiler_id = info.get("id")
    version = info.get("version")
    source = info.get("source")

    if path is None:
        return "<unresolved until configure>" if not configured else "<unknown>"

    if compiler_id and version:
        detail = "{} {}".format(compiler_id, version)
    elif compiler_id:
        detail = compiler_id
    elif version:
        detail = version
    else:
        detail = None

    if detail and source and source != "cache":
        return "{} ({}; {})".format(path, detail, source)
    if detail:
        return "{} ({})".format(path, detail)
    if source and source != "cache":
        return "{} ({})".format(path, source)
    return path

def format_python_resolution(executable: Optional[str], source: str, *, configured: bool) -> str:
    if executable is None:
        return "<unresolved until configure>" if (not configured) or source == "unresolved" else "<missing>"

    source_label = {
        "cache": "CMake cache",
        "missing": "missing",
        "unresolved": "unresolved until configure",
    }.get(source, source)
    return "{} ({})".format(executable, source_label)

def python_probe_status_text(
    available: bool,
    python_probe: Dict[str, Any],
    *,
    ok_label: str,
    unavailable_label: str,
    unresolved_label: str,
) -> str:
    if python_probe.get("status") == "unresolved":
        return unresolved_label
    return ok_label if available else unavailable_label

def collect_runtime_python_probe(context: CliContext, command: str) -> Dict[str, Any]:
    profile = context.require_profile(command)
    build_state = buildtree_state(profile.build_dir)
    cache_entries: Dict[str, Dict[str, str]] = {}
    if build_state["cmake_cache_exists"]:
        cache_entries = read_cmake_cache(profile.build_dir, command, context.profile_name())
    python_executable, python_source = doctor_python_executable(cache_entries, configured=build_state["configured"])
    python_probe = probe_python_stack(python_executable, python_source)
    python_probe["configured"] = build_state["configured"]
    return python_probe

def collect_bootstrap_state(context: CliContext, command: str) -> Dict[str, Any]:
    python_probe = collect_runtime_python_probe(context, command)
    profile = context.require_profile(command)
    cache_entries: Dict[str, Dict[str, str]] = {}
    if cmake_cache_path(profile.build_dir).exists():
        cache_entries = read_cmake_cache(profile.build_dir, command, context.profile_name())
    define_state = profile_define_state(profile, command, context.profile_name())
    mpi = collect_mpi_state(context, command, cache_entries=cache_entries, define_state=define_state)
    pre_commit = tool_probe("pre-commit", ["--version"], label="pre-commit")
    pre_commit["install_commands"] = pre_commit_install_commands() if pre_commit["status"] != "ok" else []
    plotting_install_hint = python_probe.get("install_hint")
    impacts = prerequisite_impact_entries(mpi, pre_commit, python_probe)
    return {
        "mpi": mpi,
        "pre_commit": pre_commit,
        "python": python_probe,
        "required_missing": bool(mpi["missing_required"]),
        "optional_missing": pre_commit["status"] != "ok" or not python_probe["plotting_available"],
        "plotting_install_hint": plotting_install_hint,
        "impacts": impacts,
    }

def prerequisite_impact_entries(mpi: Dict[str, Any], pre_commit: Dict[str, Any], python_probe: Dict[str, Any]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []

    if mpi.get("enabled") is not False and mpi.get("missing_required"):
        wrappers = ", ".join(str(tool) for tool in mpi["missing_required"])
        entries.append(
            {
                "name": "mpi",
                "label": "mpi wrappers unavailable ({})".format(wrappers),
                "impact": "blocks build/test/run for this MPI-enabled profile",
            }
        )

    if pre_commit.get("status") != "ok":
        entries.append(
            {
                "name": "pre_commit",
                "label": "pre-commit unavailable",
                "impact": "optional for repository hook workflows",
            }
        )

    if mpi.get("enabled") is not False and mpi.get("missing_optional"):
        launchers = ", ".join(str(tool) for tool in mpi["missing_optional"])
        entries.append(
            {
                "name": "mpi_launcher",
                "label": "mpi launcher unavailable ({})".format(launchers),
                "impact": "optional for single-rank build/test; needed for local multi-rank runs/tests",
            }
        )

    if not python_probe.get("plotting_available"):
        if python_probe.get("status") == "unresolved":
            entries.append(
                {
                    "name": "python",
                    "label": "python interpreter unresolved until configure",
                    "impact": "optional plotting only; configure the profile before checking plotting extras",
                }
            )
            return entries

        failed_modules = list(python_probe.get("failed_modules") or [])
        modules_detail = " ({})".format(", ".join(failed_modules)) if failed_modules else ""
        entries.append(
            {
                "name": "plotting",
                "label": "plotting extras unavailable{}".format(modules_detail),
                "impact": "optional plotting only",
            }
        )

    return entries

def append_prerequisite_impact_lines(lines: List[str], impacts: Sequence[Dict[str, str]], *, header: str = "Impact") -> None:
    if not impacts:
        return
    lines.append("{}:".format(header))
    for impact in impacts:
        lines.append("- {}: {}".format(impact["label"], impact["impact"]))

def build_summary_from_cache(
    profile: ProfileConfig,
    cache_entries: Dict[str, Dict[str, str]],
    define_state: Dict[str, Any],
    worktree_root: Path,
) -> Dict[str, Optional[str]]:
    requested = define_state["requested_defines"]
    mpi_requirement = resolve_mpi_requirement(worktree_root, define_state, cache_entries)
    return {
        "generator": cache_entry_value(cache_entries, "CMAKE_GENERATOR") or profile.generator,
        "build_type": cache_entry_value(cache_entries, "CMAKE_BUILD_TYPE") or requested.get("CMAKE_BUILD_TYPE"),
        "mpi": mpi_requirement["value"],
        "mpi_source": mpi_requirement["source"],
        "space_dim": cache_entry_value(cache_entries, "AMReX_SPACEDIM") or requested.get("AMReX_SPACEDIM"),
        "gpu_backend": cache_entry_value(cache_entries, "AMReX_GPU_BACKEND") or requested.get("AMReX_GPU_BACKEND"),
        "hdf5_dir": cache_entry_value(cache_entries, "HDF5_DIR", "HDF5_ROOT"),
        "hdf5_diff": cache_entry_value(cache_entries, "HDF5_DIFF_EXECUTABLE"),
        "python_enabled": cache_entry_value(cache_entries, "QUOKKA_PYTHON"),
        "python_executable": cache_entry_value(cache_entries, "_Python_EXECUTABLE", "Python_EXECUTABLE"),
    }

def doctor_python_executable(cache_entries: Dict[str, Dict[str, str]], *, configured: bool) -> Tuple[Optional[str], str]:
    cached = cache_entry_value(cache_entries, "_Python_EXECUTABLE", "Python_EXECUTABLE")
    if cached:
        return cached, "cache"
    return (None, "missing") if configured else (None, "unresolved")

def probe_python_stack(python_executable: Optional[str], source: str) -> Dict[str, Any]:
    if python_executable is None:
        status = "unresolved" if source == "unresolved" else "missing"
        return {
            "status": status,
            "executable": None,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": [],
            "modules": {},
            "install_hint": None,
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
        hint = python_install_hint({"executable": python_executable, "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"]})
        return {
            "status": "missing",
            "executable": python_executable,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
            "detail": "interpreter not found",
            "install_hint": hint,
        }
    if not probe["ok"]:
        hint = python_install_hint({"executable": python_executable, "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"]})
        return {
            "status": "error",
            "executable": python_executable,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
            "detail": first_nonempty_line(probe["stderr"], probe["stdout"]) or "probe failed",
            "install_hint": hint,
        }

    try:
        modules = json.loads(probe["stdout"]) if probe["stdout"] else {}
    except json.JSONDecodeError:
        hint = python_install_hint({"executable": python_executable, "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"]})
        return {
            "status": "error",
            "executable": python_executable,
            "source": source,
            "numpy_available": False,
            "plotting_available": False,
            "failed_modules": ["numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.cm", "PIL"],
            "modules": {},
            "detail": "invalid probe output",
            "install_hint": hint,
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

    result = {
        "status": status,
        "executable": python_executable,
        "source": source,
        "numpy_available": numpy_available,
        "plotting_available": plotting_available,
        "failed_modules": failed_modules,
        "modules": modules,
        "detail": first_error or None,
    }
    result["install_hint"] = python_install_hint(result)
    return result

def configure_fingerprint_payload(
    context: CliContext,
    command: str,
    *,
    cache_entries: Optional[Dict[str, Dict[str, str]]] = None,
    define_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = context.require_profile(command)
    payload = {
        "profile": profile.name,
        "build_dir": str(profile.build_dir),
        "generator": profile.generator,
        "executor_kind": profile.executor_kind,
        "defines": {key: profile.defines[key] for key in sorted(profile.defines)},
    }

    resolved_cache_entries = cache_entries
    if resolved_cache_entries is None and cmake_cache_path(profile.build_dir).exists():
        resolved_cache_entries = read_cmake_cache(profile.build_dir, command, context.profile_name())
    if define_state is None:
        define_state = profile_define_state(profile, command, context.profile_name())

    if resolved_cache_entries:
        build_summary = build_summary_from_cache(profile, resolved_cache_entries, define_state, context.worktree_root)
        compiler_info = compiler_metadata_from_build(profile.build_dir, resolved_cache_entries)
        payload["build"] = {
            key: build_summary.get(key)
            for key in ("generator", "build_type", "mpi", "space_dim", "gpu_backend", "hdf5_dir", "hdf5_diff", "python_enabled", "python_executable")
        }
        payload["compiler"] = {
            language: {key: compiler_info[language].get(key) for key in ("path", "id", "version")}
            for language in ("c", "cxx")
        }
    else:
        payload["env_overrides"] = {key: os.environ.get(key) for key in ENV_OVERRIDE_KEYS if os.environ.get(key)}

    return payload

def compute_configure_fingerprint(context: CliContext, command: str) -> str:
    payload = configure_fingerprint_payload(context, command)
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
