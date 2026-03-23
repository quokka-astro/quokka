from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import ConfigureRequest
from quokka.workflows.configure import run_configure


def register(app: typer.Typer) -> None:
    @app.command("configure")
    def configure_command(
        ctx: typer.Context,
        profile: Optional[str] = profile_option(),
        reconfigure: bool = typer.Option(False, "--reconfigure", help="Force CMake reconfiguration even if the build tree already looks configured.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_configure,
            ConfigureRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                reconfigure=reconfigure,
            ),
        )
