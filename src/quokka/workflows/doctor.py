from __future__ import annotations

from quokka.core.constants import LOCK_TYPES
from quokka.core.result import CommandResult
from quokka.core.types import DoctorRequest, ProfileConfig
from quokka.output.diagnostics import bootstrap_hint_command
from quokka.project.context import CliContext
from quokka.project.discovery import buildtree_state, compile_commands_path
from quokka.project.state import inspect_lock
from quokka.tools.cmake import active_env_overrides, append_prerequisite_impact_lines, build_summary_from_cache, cache_entry_value, collect_mpi_state, compiler_metadata_from_build, doctor_python_executable, format_compiler_toolchain, format_define_mismatch_summary, format_python_resolution, format_tool_probe_short, generator_tool_probe, populate_compiler_candidates_from_env, prerequisite_impact_entries, probe_python_stack, profile_define_state, python_probe_status_text, read_cmake_cache, tool_probe, toolchain_env_overrides_for_text


def run_doctor(context: CliContext, request: DoctorRequest) -> CommandResult:
    topic = request.topic or "profile"
    data: dict[str, object] = {}
    lines: list[str] = []
    profile: ProfileConfig | None = None
    configured = False
    build_state = {"cmake_cache_exists": False, "ctest_metadata_exists": False, "configured": False, "partial_configure": False}
    cache_entries: dict[str, dict[str, str]] = {}
    define_state: dict[str, object] | None = None
    python_probe: dict[str, object] | None = None

    if topic in ("all", "runtime", "profile"):
        profile = context.require_profile("doctor")
        build_state = buildtree_state(profile.build_dir)
        configured = build_state["configured"]
        define_state = profile_define_state(profile, "doctor", context.profile_name())
        if build_state["cmake_cache_exists"]:
            cache_entries = read_cmake_cache(profile.build_dir, "doctor", context.profile_name())
        python_executable, python_source = doctor_python_executable(cache_entries, configured=configured)
        python_probe = probe_python_stack(python_executable, python_source)
        python_probe["configured"] = configured

    if topic in ("all", "profile"):
        assert profile is not None
        assert define_state is not None
        assert python_probe is not None
        compile_commands = compile_commands_path(profile.build_dir)
        build_summary = build_summary_from_cache(profile, cache_entries, define_state, context.worktree_root)
        build_summary["c_compiler"] = cache_entry_value(cache_entries, "CMAKE_C_COMPILER")
        build_summary["cxx_compiler"] = cache_entry_value(cache_entries, "CMAKE_CXX_COMPILER")
        env_overrides = active_env_overrides()
        text_env_overrides = toolchain_env_overrides_for_text(env_overrides)
        mpi_state = collect_mpi_state(context, "doctor", cache_entries=cache_entries, define_state=define_state)
        compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries) if configured else {
            "c": {"path": None, "source": None, "id": None, "version": None, "metadata_path": None},
            "cxx": {"path": None, "source": None, "id": None, "version": None, "metadata_path": None},
        }
        compiler_info = populate_compiler_candidates_from_env(compiler_info, env_overrides)
        cmake_tool = tool_probe("cmake", ["--version"])
        generator_tool = generator_tool_probe(build_summary["generator"] or profile.generator)
        python_source = "cache" if cache_entry_value(cache_entries, "_Python_EXECUTABLE", "Python_EXECUTABLE") else python_probe["source"]
        toolchain = {
            "cmake": cmake_tool,
            "generator": generator_tool,
            "c": compiler_info["c"],
            "cxx": compiler_info["cxx"],
            "python": {
                "executable": build_summary["python_executable"],
                "source": python_source,
                "status": "resolved" if build_summary["python_executable"] else ("missing" if configured else "unresolved"),
            },
            "env_overrides": env_overrides,
        }
        data["profile"] = {
            "profile": context.profile_name(),
            "build_dir": str(profile.build_dir),
            "configured": configured,
            "partial_configure": build_state["partial_configure"],
            "compile_commands": str(compile_commands),
            "compile_commands_exists": compile_commands.exists(),
            "cache_path": define_state["cache_path"],
            "requested_defines": define_state["requested_defines"],
            "effective_defines": define_state["effective_defines"],
            "define_mismatches": define_state["mismatches"],
            "build": build_summary,
            "compiler": compiler_info,
            "mpi": mpi_state,
            "python": python_probe,
            "toolchain": toolchain,
        }
        if build_state["partial_configure"]:
            lines.append("profile: {} (incomplete configure)".format(context.profile_name()))
            if define_state["mismatches"]:
                lines.append("profile drift: {}".format(format_define_mismatch_summary(define_state["mismatches"])))
        elif define_state["mismatches"]:
            lines.append("profile: {} (configured, drift)".format(context.profile_name()))
            lines.append("profile drift: {}".format(format_define_mismatch_summary(define_state["mismatches"])))
        else:
            lines.append("profile: {} ({})".format(context.profile_name(), "configured" if configured else "unconfigured"))
        mpi_text = build_summary["mpi"] or "<unknown>"
        mpi_source = build_summary.get("mpi_source")
        if mpi_source and mpi_source != "CMake cache":
            mpi_text = "{} ({})".format(mpi_text, mpi_source)
        lines.append(
            "profile build: type={} mpi={} gpu={} generator={}".format(
                build_summary["build_type"] or "<unknown>",
                mpi_text,
                build_summary["gpu_backend"] or "<unknown>",
                build_summary["generator"] or profile.generator,
            )
        )
        lines.append(
            "profile configure tools: cmake={} generator={}".format(
                format_tool_probe_short(cmake_tool),
                format_tool_probe_short(generator_tool),
            )
        )
        mpi_wrappers_text = "c={} cxx={}".format(
            format_tool_probe_short(mpi_state["wrappers"]["c"]),
            format_tool_probe_short(mpi_state["wrappers"]["cxx"]),
        )
        mpi_launcher_text = format_tool_probe_short(mpi_state["launcher"])
        lines.append("profile mpi tools: {} launcher={}".format(mpi_wrappers_text, mpi_launcher_text))
        if mpi_state["missing_required"]:
            lines.append("profile mpi note: missing required MPI wrapper(s): {}".format(", ".join(mpi_state["missing_required"])))
        elif mpi_state["missing_optional"]:
            lines.append("profile mpi note: missing MPI launcher(s): {}".format(", ".join(mpi_state["missing_optional"])))
        lines.append(
            "profile toolchain: c={} cxx={}".format(
                format_compiler_toolchain(compiler_info["c"], configured=configured),
                format_compiler_toolchain(compiler_info["cxx"], configured=configured),
            )
        )
        lines.append(
            "profile python: {}".format(
                format_python_resolution(build_summary["python_executable"], python_source, configured=configured)
            )
        )
        if text_env_overrides:
            lines.append("profile overrides: {}".format(", ".join("{}={}".format(key, text_env_overrides[key]) for key in sorted(text_env_overrides))))
        if build_state["partial_configure"]:
            lines.append("profile configure: CMake cache exists but CTest metadata is missing")
        lines.append("compile_commands: {} ({})".format("present" if compile_commands.exists() else "missing", compile_commands))
        if build_summary["hdf5_dir"]:
            lines.append("profile deps: hdf5={}".format(build_summary["hdf5_dir"]))

    if topic in ("all", "runtime"):
        runtime_dir = context.resolve_runtime_dir("doctor", create=False)
        state_db_path = context.db_path("doctor", create=False)
        assert profile is not None
        assert python_probe is not None
        mpi_state = collect_mpi_state(context, "doctor", cache_entries=cache_entries, define_state=define_state)
        tools = {
            "cmake": tool_probe("cmake", ["--version"]),
            "ctest": tool_probe("ctest", ["--version"]),
            "git": tool_probe("git", ["--version"]),
            "generator": generator_tool_probe(profile.generator),
            "pre_commit": tool_probe("pre-commit", ["--version"], label="pre-commit"),
        }
        data["runtime"] = {
            "runtime_dir": str(runtime_dir),
            "state_db": str(state_db_path),
            "state_db_exists": state_db_path.exists(),
            "sqlite_ok": True,
            "tools": tools,
            "mpi": mpi_state,
            "python": python_probe,
            "impacts": prerequisite_impact_entries(mpi_state, tools["pre_commit"], python_probe),
        }
        lines.append("runtime: ok ({})".format(runtime_dir))
        lines.append(
            "tools: cmake={cmake}, ctest={ctest}, git={git}, {generator_label}={generator_status}, mpi={mpi}, pre-commit={pre_commit}".format(
                cmake=tools["cmake"]["status"],
                ctest=tools["ctest"]["status"],
                git=tools["git"]["status"],
                generator_label=profile.generator,
                generator_status=tools["generator"]["status"],
                mpi=mpi_state["status"],
                pre_commit=tools["pre_commit"]["status"],
            )
        )
        if mpi_state["setting"] is not None:
            lines.append("mpi: setting={} ({})".format(mpi_state["setting"], mpi_state["source"]))
        lines.append(
            "mpi tools: c={} cxx={} launcher={}".format(
                format_tool_probe_short(mpi_state["wrappers"]["c"]),
                format_tool_probe_short(mpi_state["wrappers"]["cxx"]),
                format_tool_probe_short(mpi_state["launcher"]),
            )
        )
        mpi_hint = mpi_state.get("install_hint")
        if isinstance(mpi_hint, str) and mpi_hint:
            lines.append("mpi hint: {}".format(mpi_hint))
        plotting_state = python_probe_status_text(
            python_probe["plotting_available"],
            python_probe,
            ok_label="ok",
            unavailable_label="unavailable",
            unresolved_label="unresolved until configure",
        )
        plotting_detail = ""
        if python_probe["failed_modules"]:
            plotting_detail = " ({})".format(", ".join(python_probe["failed_modules"]))
        interpreter_text = format_python_resolution(
            python_probe["executable"],
            str(python_probe.get("source") or "missing"),
            configured=configured,
        )
        lines.append(
            "python: interpreter={} numpy={} plotting={}{}".format(
                interpreter_text,
                python_probe_status_text(
                    python_probe["numpy_available"],
                    python_probe,
                    ok_label="ok",
                    unavailable_label="missing",
                    unresolved_label="unresolved until configure",
                ),
                plotting_state,
                plotting_detail,
            )
        )
        install_hint = python_probe.get("install_hint")
        if python_probe["status"] == "unresolved":
            lines.append("python note: configure the profile first to resolve the CMake-selected interpreter.")
        elif isinstance(install_hint, str) and install_hint and not python_probe["plotting_available"]:
            lines.append("python hint: install plotting extras with {}".format(install_hint))
        append_prerequisite_impact_lines(lines, prerequisite_impact_entries(mpi_state, tools["pre_commit"], python_probe))
        if mpi_state["status"] != "ok" or tools["pre_commit"]["status"] != "ok" or not python_probe["plotting_available"]:
            lines.append("bootstrap hint: {}".format(bootstrap_hint_command(context.profile_name())))

    if topic in ("all", "locking"):
        locks = []
        for lock_type in LOCK_TYPES:
            info = inspect_lock(context, lock_type, "doctor", probe_active=False)
            locks.append({"lock_type": lock_type, "active": info.active, "metadata": info.metadata})
        data["locking"] = {"locks": locks}
        lines.append("locking: {}".format(", ".join("{}={}".format(lock["lock_type"], "active" if lock["active"] else "idle") for lock in locks)))

    return CommandResult("doctor", context.profile_name(), None, data, "\n".join(lines))
