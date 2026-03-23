from __future__ import annotations

import contextlib
import unittest
from pathlib import Path
from unittest.mock import patch

from quokka.core.types import ConfigureRequest, ProfileConfig
from quokka.workflows.configure import run_configure


class FakeContext:
    def __init__(self) -> None:
        self.profile = ProfileConfig(
            name="host-3d-release",
            build_dir=Path("/tmp/build"),
            generator="Ninja",
            defines={},
            executor_kind="local",
            executor={},
        )

    def require_profile(self, command: str) -> ProfileConfig:
        return self.profile

    def profile_name(self) -> str:
        return self.profile.name

    def resolve_runtime_dir(self, command: str) -> Path:
        return Path("/tmp/runtime")

    def open_db(self, command: str) -> object:
        return object()


class WorkflowConfigureTest(unittest.TestCase):
    def test_run_configure_formats_result(self) -> None:
        request = ConfigureRequest(reconfigure=False)
        with patch("quokka.workflows.configure.acquire_lock", return_value=contextlib.nullcontext()):
            with patch("quokka.workflows.configure.perform_configure", return_value={"configure_fingerprint": "abc123"}):
                result = run_configure(FakeContext(), request)
        self.assertEqual(result.command, "configure")
        self.assertIn("Configured profile", result.text)


if __name__ == "__main__":
    unittest.main()
