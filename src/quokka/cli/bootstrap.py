from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import BootstrapRequest
from quokka.workflows.bootstrap import run_bootstrap

BOOTSTRAP_HELP = """Check or install developer prerequisites for the selected profile.

Examples:
  quokka bootstrap --profile host-3d-release
  quokka bootstrap --profile host-3d-release --fix
  quokka bootstrap --profile host-3d-release --fix --include-optional
"""


def register(app: typer.Typer) -> None:
    @app.command("bootstrap", help=BOOTSTRAP_HELP)
    def bootstrap_command(
        ctx: typer.Context,
        profile: Optional[str] = profile_option(),
        fix: bool = typer.Option(False, "--fix", help="Install missing required prerequisites when possible.", is_flag=True),
        include_optional: bool = typer.Option(False, "--include-optional", help="Also install optional plotting extras when the configured Python interpreter is known.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_bootstrap,
            BootstrapRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                fix=fix,
                include_optional=include_optional,
            ),
        )
