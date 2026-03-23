from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import LockRequest
from quokka.workflows.locking import run_lock


class LockAction(str, Enum):
    ls = "ls"
    break_ = "break"


def register(app: typer.Typer) -> None:
    @app.command("lock")
    def lock_command(
        ctx: typer.Context,
        lock_action: LockAction = typer.Argument(..., help="Lock action to perform."),
        profile: Optional[str] = profile_option(),
        scope: Optional[str] = typer.Option(None, "--scope", help="Reserved lock scope override."),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_lock,
            LockRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                lock_action="break" if lock_action is LockAction.break_ else lock_action.value,
                scope=scope,
            ),
        )
