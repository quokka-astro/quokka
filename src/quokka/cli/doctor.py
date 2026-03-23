from __future__ import annotations

from enum import Enum
from typing import Optional

import typer

from quokka.cli.common import execute_request, json_option, profile_option, worktree_from_context
from quokka.core.types import DoctorRequest
from quokka.workflows.doctor import run_doctor


class DoctorTopic(str, Enum):
    all = "all"
    locking = "locking"
    runtime = "runtime"
    profile = "profile"


def register(app: typer.Typer) -> None:
    @app.command("doctor")
    def doctor_command(
        ctx: typer.Context,
        topic: DoctorTopic = typer.Argument(DoctorTopic.profile, help="Diagnostic topic."),
        profile: Optional[str] = profile_option(),
        json_output: bool = json_option(),
    ) -> None:
        execute_request(
            run_doctor,
            DoctorRequest(
                worktree=worktree_from_context(ctx),
                profile=profile,
                json_output=json_output,
                topic=topic.value,
            ),
        )
