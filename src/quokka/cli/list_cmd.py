from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import ListRequest
from quokka.workflows.list_items import run_list


class ListKind(str, Enum):
    problems = "problems"
    tests = "tests"
    profiles = "profiles"


def register(app: typer.Typer) -> None:
    @app.command("list")
    def list_command(
        ctx: typer.Context,
        list_kind: ListKind = typer.Argument(..., help="Resource type to list."),
        profile: Optional[str] = profile_option(),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_list,
            ListRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                list_kind=list_kind.value,
            ),
        )
