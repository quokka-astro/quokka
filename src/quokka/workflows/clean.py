from __future__ import annotations

import shutil

from quokka.core.types import CleanRequest, CommandResult
from quokka.project.context import CliContext
from quokka.project.state import artifact_receipts_dir, break_locks, configure_receipt_path


def run_clean(context: CliContext, request: CleanRequest) -> CommandResult:
    action = request.clean_kind
    if action == "runs":
        runs_dir = context.resolve_runtime_dir("clean") / "runs" / "wt-{}".format(context.worktree_id)
        existed = runs_dir.exists()
        if existed:
            shutil.rmtree(runs_dir)
        data = {"removed": str(runs_dir), "existed": existed}
        return CommandResult("clean", context.profile_name(), None, data, "Cleaned run scratch state.")

    if action == "locks":
        broken = break_locks(context, "clean")
        return CommandResult("clean", context.profile_name(), None, {"broken": broken}, "Removed {} stale lock(s).".format(len(broken)))

    profile = context.require_profile("clean")
    removed: list[str] = []
    for path in sorted(artifact_receipts_dir(profile.build_dir).glob("*.json")):
        path.unlink()
        removed.append(str(path))
    configure_path = configure_receipt_path(profile.build_dir)
    if configure_path.exists():
        configure_path.unlink()
        removed.append(str(configure_path))
    data = {"removed": removed}
    text = "Removed {} receipt file(s) for profile {}.".format(len(removed), context.profile_name())
    return CommandResult("clean", context.profile_name(), None, data, text)
