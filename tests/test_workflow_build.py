from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from quokka.core.types import BuildRequest, ProfileConfig
from quokka.workflows.build import run_build


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

    def profile_name(self) -> str:
        return self.profile.name


class WorkflowBuildTest(unittest.TestCase):
    def test_run_build_formats_result(self) -> None:
        request = BuildRequest(targets=("HydroWave",), reconfigure=False)
        with patch("quokka.workflows.build.perform_build", return_value={"build_dir": "/tmp/build", "receipts_written": ["HydroWave"]}):
            result = run_build(FakeContext(), request)
        self.assertEqual(result.command, "build")
        self.assertIn("HydroWave", result.text)


if __name__ == "__main__":
    unittest.main()
