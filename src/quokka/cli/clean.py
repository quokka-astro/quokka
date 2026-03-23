from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import CleanRequest
from quokka.workflows.clean import run_clean


class CleanKind(str, Enum):
    runs = "runs"
    locks = "locks"
    profile = "profile"


def register(app: typer.Typer) -> None:
    @app.command("clean")
    def clean_command(
        ctx: typer.Context,
        clean_kind: CleanKind = typer.Argument(..., help="State category to clean."),
        profile: Optional[str] = profile_option(),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_clean,
            CleanRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                clean_kind=clean_kind.value,
            ),
        )
