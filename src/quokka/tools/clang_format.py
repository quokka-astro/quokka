from __future__ import annotations

import shutil
from typing import Sequence

from quokka.core.errors import DiagnosticError
from quokka.core.subprocess import run_command
from quokka.project.context import CliContext

def clang_format_files(context: CliContext, files: Sequence[str]) -> list[str]:
    formatter = shutil.which("clang-format")
    if formatter is None:
        raise DiagnosticError(
            "EXECUTOR_UNAVAILABLE",
            "clang-format is required but not installed or not available on PATH.",
            command="format",
        )

    formatted_files = list(files)
    if not formatted_files:
        return []

    run_command(
        [formatter, "-i", "-style=file"] + formatted_files,
        cwd=context.worktree_root,
        command="format",
        profile=None,
        capture_output=context.json_output,
    )
    return formatted_files
