from __future__ import annotations

from typing import List, Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import BuildRequest
from quokka.workflows.build import run_build


def register(app: typer.Typer) -> None:
    @app.command("build")
    def build_command(
        ctx: typer.Context,
        targets: Optional[List[str]] = typer.Argument(None, help="Optional build target names."),
        profile: Optional[str] = profile_option(),
        reconfigure: bool = typer.Option(False, "--reconfigure", help="Re-run CMake configure before building.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_build,
            BuildRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                targets=tuple(targets or ()),
                reconfigure=reconfigure,
            ),
        )
