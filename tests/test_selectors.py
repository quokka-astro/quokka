from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from quokka.core.types import ProfileConfig, TestRequest, TestSpec
from quokka.model.selectors import ctest_selection, resolve_format_selector, resolve_tidy_selector


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


class SelectorsTest(unittest.TestCase):
    def test_selector_validation(self) -> None:
        self.assertEqual(resolve_tidy_selector("changed", profile="host-3d-release"), "changed")
        self.assertEqual(resolve_format_selector("all"), "all")

    def test_ctest_selection_exact_name(self) -> None:
        tests = [
            TestSpec(name="ODEIntegration", command=["ODEIntegration"], working_directory=None, source_path=Path("CTestTestfile.cmake")),
            TestSpec(name="HydroWave", command=["HydroWave"], working_directory=None, source_path=Path("CTestTestfile.cmake")),
        ]
        request = TestRequest(test_name="HydroWave", ctest_regex=None)
        with patch("quokka.model.selectors.discover_tests", return_value=tests):
            selected = ctest_selection(FakeContext(), request)
        self.assertEqual([test.name for test in selected], ["HydroWave"])


if __name__ == "__main__":
    unittest.main()
