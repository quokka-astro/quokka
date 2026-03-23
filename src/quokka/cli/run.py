from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import RunRequest
from quokka.workflows.run import run_problem


def register(app: typer.Typer) -> None:
    @app.command("run")
    def run_command(
        ctx: typer.Context,
        problem: str = typer.Argument(..., help="Problem executable name."),
        profile: Optional[str] = profile_option(),
        input_path: Optional[str] = typer.Option(None, "--input", help="Input file path."),
        build_if_needed: bool = typer.Option(False, "--build-if-needed", help="Build the problem if receipts are missing or stale.", is_flag=True),
        verbose_runtime: bool = typer.Option(False, "--verbose-runtime", help="Show the full stdout/stderr emitted by the executable.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_problem,
            RunRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                problem=problem,
                input=input_path,
                build_if_needed=build_if_needed,
                verbose_runtime=verbose_runtime,
            ),
        )
