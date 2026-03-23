from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, worktree_from_context
from quokka.core.types import FormatRequest
from quokka.workflows.format import run_format

FORMAT_HELP = """Run clang-format directly on selected C/C++ files.

Selectors:
  changed   Files modified in the working tree relative to HEAD (default)
  previous  Files modified in the previous commit
  origin    Files different from origin/<current-branch>
  dev       Files different from the local development branch
  all       All tracked C/C++ files supported by clang-format
"""


def register(app: typer.Typer) -> None:
    @app.command("format", help=FORMAT_HELP)
    def format_command(
        ctx: typer.Context,
        selector: Optional[str] = typer.Argument(None, help="File selector: changed (default), previous, origin, dev, or all."),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_format,
            FormatRequest(
                worktree=worktree_from_context(ctx),
                json_output=json_output,
                selector=selector,
            ),
        )
