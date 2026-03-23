from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import StatusRequest
from quokka.workflows.status import run_status


def register(app: typer.Typer) -> None:
    @app.command("status")
    def status_command(
        ctx: typer.Context,
        profile: Optional[str] = profile_option(),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_status,
            StatusRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
            ),
        )
