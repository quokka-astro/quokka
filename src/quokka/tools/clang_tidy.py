
from __future__ import annotations

from quokka.core.subprocess import run_command
from quokka.project.context import CliContext


def run_clang_tidy(context: CliContext, selector: str, *, fix: bool) -> None:
    profile = context.require_profile('tidy')
    script = context.worktree_root / 'scripts' / 'bash' / 'tidy.sh'
    command = [str(script)]
    if fix:
        command.append('--fix')
    command.extend([str(profile.build_dir), selector])
    run_command(command, cwd=context.worktree_root, command='tidy', profile=context.profile_name(), capture_output=context.json_output)
