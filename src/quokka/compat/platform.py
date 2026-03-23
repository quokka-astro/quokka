
from __future__ import annotations

import os


def executable_name(name: str) -> str:
    return name + '.exe' if os.name == 'nt' else name
