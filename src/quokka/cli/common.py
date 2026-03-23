from __future__ import annotations

import os
import traceback
from typing import Any, Callable, Optional, TypeVar

import typer

from quokka.core.errors import DiagnosticError
from quokka.core.result import CommandResult
from quokka.core.types import CommandRequest
from quokka.output.diagnostics import error_result
from quokka.output.summary import format_result
from quokka.project.context import CliContext, context_for_request

RequestT = TypeVar("RequestT", bound=CommandRequest)
WorkflowHandler = Callable[[CliContext, RequestT], CommandResult]


def profile_option() -> Any:
    return typer.Option(None, "--profile", help="Profile name from quokka.toml.")


def json_option() -> Any:
    return typer.Option(False, "--json", help="Emit machine-readable JSON.", is_flag=True)


def worktree_option() -> Any:
    return typer.Option(None, "-C", "--worktree", help="Path to the target worktree.")


def worktree_from_context(ctx: typer.Context) -> Optional[str]:
    state = ctx.obj or {}
    worktree = state.get("worktree")
    return None if worktree is None else str(worktree)


def execute_request(handler: WorkflowHandler[RequestT], request: RequestT) -> None:
    context: CliContext | None = None
    try:
        context = context_for_request(request)
        result = handler(context, request)
        typer.echo(format_result(result, request.json_output))
    except DiagnosticError as exc:
        effective_profile = exc.profile or (context.profile_name() if context is not None else request.profile)
        output = error_result(exc, request.json_output, request.command_name, effective_profile)
        typer.echo(output, err=not request.json_output)
        raise typer.Exit(code=exc.exit_code) from exc
    except Exception as exc:
        if os.environ.get("QUOKKA_DEBUG"):
            traceback.print_exc()
        effective_profile = context.profile_name() if context is not None else request.profile
        error = DiagnosticError(
            "INTERNAL_ERROR",
            "Unexpected CLI failure: {}".format(exc),
            command=request.command_name,
            profile=effective_profile,
        )
        output = error_result(error, request.json_output, request.command_name, effective_profile)
        typer.echo(output, err=not request.json_output)
        raise typer.Exit(code=error.exit_code) from exc
