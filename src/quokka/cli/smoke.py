from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import SmokeRequest
from quokka.workflows.smoke import run_smoke


def register(app: typer.Typer) -> None:
    @app.command("smoke")
    def smoke_command(
        ctx: typer.Context,
        test_name: Optional[str] = typer.Argument(None, help="Optional smoke-test override."),
        profile: Optional[str] = profile_option(),
        stream: bool = typer.Option(False, "--stream", help="Stream live test progress and stdout/stderr; repetitive timestep banners are throttled.", is_flag=True),
        compact_stream: bool = typer.Option(False, "--compact-stream", help="Show compact live progress and write the full log to a file.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_smoke,
            SmokeRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                test_name=test_name,
                stream=stream,
                compact_stream=compact_stream,
            ),
        )
