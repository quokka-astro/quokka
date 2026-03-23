from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import TidyRequest
from quokka.workflows.tidy import run_tidy

TIDY_HELP = """Run clang-tidy on files selected relative to Git history.

Selectors:
  changed   Files modified in the working tree relative to HEAD (default)
  previous  Files modified in the previous commit
  origin    Files different from origin/<current-branch>
  dev       Files different from the local development branch
"""


def register(app: typer.Typer) -> None:
    @app.command("tidy", help=TIDY_HELP)
    def tidy_command(
        ctx: typer.Context,
        selector: Optional[str] = typer.Argument(None, help="File selector: changed (default), previous, origin, or dev."),
        profile: Optional[str] = profile_option(),
        fix: bool = typer.Option(False, "--fix", help="Apply clang-tidy fix-it hints.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_tidy,
            TidyRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                selector=selector,
                fix=fix,
            ),
        )
