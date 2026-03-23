from __future__ import annotations

import shlex

from quokka.core.subprocess import run_command, shell_join
from quokka.core.result import CommandResult
from quokka.core.types import BootstrapRequest
from quokka.output.console import emit_notice
from quokka.output.diagnostics import bootstrap_hint_command
from quokka.project.context import CliContext
from quokka.tools.cmake import append_prerequisite_impact_lines, collect_bootstrap_state, preferred_pre_commit_install_command, python_probe_status_text


def run_bootstrap(context: CliContext, request: BootstrapRequest) -> CommandResult:
    state = collect_bootstrap_state(context, "bootstrap")
    mpi = state["mpi"]
    pre_commit = state["pre_commit"]
    python_probe = state["python"]
    installed: list[str] = []
    skipped: list[str] = []

    if request.fix:
        if mpi["status"] in {"missing", "partial", "unknown"}:
            skipped.append("MPI toolchain (install or load manually)")
        if pre_commit["status"] != "ok":
            install_command = preferred_pre_commit_install_command()
            if install_command is None:
                skipped.append("pre-commit (no usable installer command found)")
            else:
                emit_notice(context, "Installing pre-commit with: {}".format(shell_join(install_command)))
                run_command(install_command, command="bootstrap", profile=context.profile_name(), capture_output=context.json_output)
                installed.append("pre-commit")

        if request.include_optional and not python_probe["plotting_available"]:
            if python_probe["status"] == "unresolved":
                skipped.append("plotting extras (configure the profile first to resolve the interpreter)")
            else:
                install_hint = state.get("plotting_install_hint")
                if isinstance(install_hint, str) and install_hint:
                    install_command = shlex.split(install_hint)
                    emit_notice(context, "Installing optional plotting extras with: {}".format(shell_join(install_command)))
                    run_command(install_command, command="bootstrap", profile=context.profile_name(), capture_output=context.json_output)
                    installed.append("plotting extras")
                else:
                    skipped.append("plotting extras (no install hint available)")

        state = collect_bootstrap_state(context, "bootstrap")
        mpi = state["mpi"]
        pre_commit = state["pre_commit"]
        python_probe = state["python"]

    data = {
        "profile": context.profile_name(),
        "mpi": mpi,
        "pre_commit": pre_commit,
        "python": python_probe,
        "impacts": state["impacts"],
        "installed": installed,
        "skipped": skipped,
    }

    lines = ["Bootstrap status for profile {}.".format(context.profile_name())]
    mpi_required_state = mpi["status"]
    if mpi["missing_required"]:
        mpi_required_state += " ({})".format(", ".join(mpi["missing_required"]))
    lines.append("Required: mpi={}".format(mpi_required_state))
    if mpi["setting"] is not None:
        lines.append("MPI setting: {} ({})".format(mpi["setting"], mpi["source"]))
    launcher_state = "ok" if mpi["launcher"]["status"] == "ok" else "missing ({})".format(mpi["launcher"]["tool"])
    pre_commit_state = pre_commit["status"]
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
    lines.append("Optional: pre-commit={}, mpi launcher={}, plotting={}{}".format(pre_commit_state, launcher_state, plotting_state, plotting_detail))
    append_prerequisite_impact_lines(lines, state["impacts"])

    if installed:
        lines.append("Installed: {}".format(", ".join(installed)))
    if skipped:
        lines.append("Skipped: {}".format(", ".join(skipped)))

    next_steps: list[str] = []
    mpi_hint = mpi.get("install_hint")
    if isinstance(mpi_hint, str) and mpi_hint:
        next_steps.append(mpi_hint)
    if pre_commit["status"] != "ok":
        next_steps.append("Install pre-commit hooks tooling with: {}".format(bootstrap_hint_command(context.profile_name(), fix=True)))
    if not python_probe["plotting_available"]:
        if python_probe["status"] == "unresolved":
            next_steps.append("Configure the profile first, then rerun {}.".format(bootstrap_hint_command(context.profile_name(), fix=True, include_optional=True)))
        else:
            next_steps.append(bootstrap_hint_command(context.profile_name(), fix=True, include_optional=True))
    if next_steps:
        lines.append("Next steps:")
        for step in next_steps:
            lines.append("- {}".format(step))
    else:
        lines.append("All checked prerequisites are ready.")

    return CommandResult("bootstrap", context.profile_name(), None, data, "\n".join(lines))
