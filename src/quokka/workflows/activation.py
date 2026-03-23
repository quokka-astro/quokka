from __future__ import annotations

import shlex

from quokka.core.types import ActivationRequest, CommandResult
from quokka.project.context import CliContext


def run_activation_env(context: CliContext, request: ActivationRequest) -> CommandResult:
    profile = context.require_profile(request.command_name)
    runtime_dir = context.resolve_runtime_dir(request.command_name)
    prompt = "(quokka:{}@{})".format(context.worktree_root.name, profile.name)
    exports = {
        "QUOKKA_ACTIVE": "1",
        "QUOKKA_WORKTREE_ROOT": str(context.worktree_root),
        "QUOKKA_WORKTREE_ID": context.worktree_id,
        "QUOKKA_PROFILE": profile.name,
        "QUOKKA_RUNTIME_DIR": str(runtime_dir),
        "QUOKKA_PROMPT_PREFIX": prompt,
    }
    lines = ["export {}={}".format(key, shlex.quote(value)) for key, value in exports.items()]
    return CommandResult(request.command_name, profile.name, None, {"exports": exports}, "\n".join(lines))
