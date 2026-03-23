from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quokka.model.tests import parse_ctest_testfiles


class ModelTestsTest(unittest.TestCase):
    def test_parse_ctest_testfiles_applies_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            problem_dir = build_dir / "src" / "problems" / "HydroWave"
            problem_dir.mkdir(parents=True)

            (build_dir / "CTestTestfile.cmake").write_text('subdirs("src/problems/HydroWave")\n', encoding="utf-8")
            (problem_dir / "CTestTestfile.cmake").write_text(
                "\n".join(
                    [
                        'add_test(HydroWave "/tmp/HydroWave" "../inputs/HydroWave.in")',
                        'set_tests_properties(HydroWave PROPERTIES WORKING_DIRECTORY "/tmp/run-dir")',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            tests = parse_ctest_testfiles(build_dir, "status", "host-3d-release")

        self.assertEqual(len(tests), 1)
        self.assertEqual(tests[0].name, "HydroWave")
        self.assertEqual(tests[0].working_directory, Path("/tmp/run-dir").resolve())


if __name__ == "__main__":
    unittest.main()
