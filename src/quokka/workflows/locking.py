from __future__ import annotations

from quokka.core.constants import LOCK_TYPES
from quokka.core.result import CommandResult
from quokka.core.types import LockRequest
from quokka.project.context import CliContext
from quokka.project.state import break_locks, inspect_lock


def run_lock(context: CliContext, request: LockRequest) -> CommandResult:
    if request.lock_action == "ls":
        locks = []
        for lock_type in LOCK_TYPES:
            info = inspect_lock(context, lock_type, "lock", probe_active=False)
            locks.append(
                {
                    "lock_type": lock_type,
                    "active": info.active,
                    "lock_path": str(info.lock_path),
                    "metadata_path": str(info.metadata_path),
                    "metadata": info.metadata,
                }
            )
        text = "\n".join("{}: {}".format(lock["lock_type"], "active" if lock["active"] else "idle") for lock in locks)
        return CommandResult("lock", context.profile_name(), None, {"locks": locks}, text)

    broken = break_locks(context, "lock")
    text = "Removed {} stale lock(s).".format(len(broken))
    return CommandResult("lock", context.profile_name(), None, {"broken": broken}, text)
