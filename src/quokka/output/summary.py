from __future__ import annotations

import json

from quokka.core.result import CommandResult
from quokka.output.json import success_payload


def format_result(result: CommandResult, as_json: bool) -> str:
    if not as_json:
        return result.text
    return json.dumps(success_payload(result), indent=2, sort_keys=True)
