from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from quokka.core.constants import DIAGNOSTIC_CODES

class DiagnosticError(RuntimeError):
    def __init__(
        self,
        diagnostic_id: str,
        message: str,
        *,
        command: Optional[str] = None,
        profile: Optional[str] = None,
        resource: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_id = diagnostic_id
        self.exit_code = DIAGNOSTIC_CODES[diagnostic_id]
        self.command = command
        self.profile = profile
        self.resource = resource
        self.details = details or {}
