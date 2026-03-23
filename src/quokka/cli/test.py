from __future__ import annotations

from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import TestRequest
from quokka.workflows.test import run_test


def register(app: typer.Typer) -> None:
    @app.command("test")
    def test_command(
        ctx: typer.Context,
        test_name: Optional[str] = typer.Argument(None, help="Exact CTest name to run."),
        profile: Optional[str] = profile_option(),
        ctest_regex: Optional[str] = typer.Option(None, "--ctest-regex", help="CTest regex selector."),
        build_if_needed: bool = typer.Option(False, "--build-if-needed", help="Configure/build required test targets before running CTest.", is_flag=True),
        stream: bool = typer.Option(False, "--stream", help="Stream live test progress and stdout/stderr; repetitive timestep banners are throttled.", is_flag=True),
        compact_stream: bool = typer.Option(False, "--compact-stream", help="Show compact live progress and write the full log to a file.", is_flag=True),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_test,
            TestRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                test_name=test_name,
                ctest_regex=ctest_regex,
                build_if_needed=build_if_needed,
                stream=stream,
                compact_stream=compact_stream,
            ),
        )
