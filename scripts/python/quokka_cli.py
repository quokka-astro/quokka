#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from quokka.cli.app import main as package_main

    return package_main()


if __name__ == "__main__":
    raise SystemExit(main())
