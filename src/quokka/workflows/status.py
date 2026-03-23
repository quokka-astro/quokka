from __future__ import annotations

from quokka.core.constants import LOCK_TYPES
from quokka.core.result import CommandResult
from quokka.core.types import StatusRequest
from quokka.model.files import default_input_for_problem
from quokka.model.targets import discover_problems
from quokka.project.context import CliContext
from quokka.project.discovery import buildtree_state
from quokka.project.state import configure_receipt_matches_current_build, configure_receipt_path, inspect_lock, read_json, state_for_artifact
from quokka.tools.cmake import build_summary_from_cache, compiler_metadata_from_build, format_define_mismatch_summary, profile_define_state, read_cmake_cache


def run_status(context: CliContext, request: StatusRequest) -> CommandResult:
    profile = context.require_profile("status")
    runtime_dir = context.resolve_runtime_dir("status", create=False)
    build_state = buildtree_state(profile.build_dir)

    locks = []
    for lock_type in LOCK_TYPES:
        info = inspect_lock(context, lock_type, "status", probe_active=False)
        locks.append(
            {
                "lock_type": lock_type,
                "active": info.active,
                "metadata_path": str(info.metadata_path),
                "metadata": info.metadata,
            }
        )

    configured = build_state["configured"]
    configure_receipt = configure_receipt_path(profile.build_dir)
    define_state = profile_define_state(profile, "status", context.profile_name())
    cache_entries = read_cmake_cache(profile.build_dir, "status", context.profile_name()) if build_state["cmake_cache_exists"] else {}
    build_summary = build_summary_from_cache(profile, cache_entries, define_state, context.worktree_root)
    compiler_info = compiler_metadata_from_build(profile.build_dir, cache_entries) if configured else {
        "c": {"path": None, "id": None, "version": None, "metadata_path": None},
        "cxx": {"path": None, "id": None, "version": None, "metadata_path": None},
    }
    configure_state = None
    if configure_receipt.exists():
        configure_data = read_json(configure_receipt, "status", context.profile_name())
        receipt_state = "ready" if configured and configure_receipt_matches_current_build(configure_data, build_summary, compiler_info) else "stale"
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
    elif build_state["partial_configure"]:
        configure_state = {
            "receipt_path": str(configure_receipt),
            "configured_at": None,
            "state": "incomplete",
            "cache_path": define_state["cache_path"],
            "requested_defines": define_state["requested_defines"],
            "effective_defines": define_state["effective_defines"],
            "define_mismatches": define_state["mismatches"],
            "compiler": compiler_info,
            "build": build_summary,
        }

    problem_states: dict[str, int] = {"ready": 0, "missing": 0, "stale_source": 0, "stale_configure": 0, "unknown": 0}
    artifact_summary = {"ready": 0, "not_built": 0, "stale_source": 0, "stale_configure": 0, "unknown": 0}
    problem_examples: dict[str, list[dict[str, str]]] = {state: [] for state in problem_states}
    repair_hints: dict[str, str] = {}
    for problem in discover_problems(profile.build_dir, "status", context.profile_name()) if configured else []:
        default_input = default_input_for_problem(context, problem, "status")
        state, details = state_for_artifact(context, problem, "status", default_input)
        problem_states[state] = problem_states.get(state, 0) + 1
        summary_state = "not_built" if state == "missing" else state
        artifact_summary[summary_state] = artifact_summary.get(summary_state, 0) + 1
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
        "runtime_dir": str(runtime_dir),
        "build_dir": str(profile.build_dir),
        "configured": configured,
        "partial_configure": build_state["partial_configure"],
        "configure": configure_state,
        "locks": locks,
        "artifacts": problem_states,
        "artifact_summary": artifact_summary,
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
        "artifacts: ready={ready} not_built={not_built} stale_source={stale_source} stale_configure={stale_configure} unknown={unknown}".format(
            **artifact_summary
        ),
    ]
    if artifact_summary["not_built"] > 0:
        lines.append("not_built means the problem is known for this profile but has not been compiled yet.")
    if build_state["partial_configure"]:
        lines.append("configure: incomplete (CMake cache exists but CTest metadata is missing)")
        if configure_state is not None and configure_state.get("define_mismatches"):
            lines.append("configure drift: {}".format(format_define_mismatch_summary(configure_state["define_mismatches"])))
    elif configure_state is not None and configure_state.get("define_mismatches"):
        lines.append("configure: drift ({})".format(format_define_mismatch_summary(configure_state["define_mismatches"])))
    for state in ("missing", "stale_source", "stale_configure", "unknown"):
        examples = problem_examples.get(state) or []
        if not examples:
            continue
        hidden = problem_states.get(state, 0) - len(examples)
        suffix = " (+{} more)".format(hidden) if hidden > 0 else ""
        label = "not_built" if state == "missing" else state
        lines.append("{} examples: {}{}".format(label, ", ".join(example["name"] for example in examples), suffix))
        hint = repair_hints.get(state)
        if hint:
            lines.append("{} repair: {}".format(label, hint))
    return CommandResult("status", context.profile_name(), None, data, "\n".join(lines))
