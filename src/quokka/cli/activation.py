from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import ActivationRequest
from quokka.workflows.activation import run_activation_env


def register(app: typer.Typer) -> None:
    @app.command("_activate-env", hidden=True)
    def activation_command(
        ctx: typer.Context,
        profile: Optional[str] = profile_option(),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_activation_env,
            ActivationRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
            ),
        )
