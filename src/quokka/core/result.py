from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CommandResult:
    command: str
    profile: str | None
    resource: dict[str, Any] | None
    data: dict[str, Any]
    text: str
