
from __future__ import annotations

from typing import Any, Dict, Optional

from quokka.core.errors import DiagnosticError
from quokka.core.result import CommandResult


def success_payload(result: CommandResult) -> Dict[str, Any]:
    return {
        'schema': 1,
        'ok': True,
        'command': result.command,
        'profile': result.profile,
        'resource': result.resource,
        'diagnostic': None,
        'data': result.data,
    }


def error_payload(error: DiagnosticError, command: Optional[str], profile: Optional[str], hints: list[str]) -> Dict[str, Any]:
    return {
        'schema': 1,
        'ok': False,
        'command': error.command or command,
        'profile': error.profile or profile,
        'resource': error.resource,
        'diagnostic': {
            'id': error.diagnostic_id,
            'exit_code': error.exit_code,
            'message': error.args[0],
            'details': error.details,
            'hints': hints,
        },
    }
