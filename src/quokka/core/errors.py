from __future__ import annotations

from typing import Any, Optional

from quokka.core.constants import DIAGNOSTIC_CODES


class DiagnosticError(RuntimeError):
    def __init__(
        self,
        diagnostic_id: str,
        message: str,
        *,
        command: Optional[str] = None,
        profile: Optional[str] = None,
        resource: Optional[dict[str, Any]] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_id = diagnostic_id
        self.exit_code = DIAGNOSTIC_CODES[diagnostic_id]
        self.command = command
        self.profile = profile
        self.resource = resource
        self.details = details or {}
