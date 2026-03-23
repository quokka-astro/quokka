from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quokka.tools.cmake import cmake_bool_state, read_cmake_cache


class CMakeTest(unittest.TestCase):
    def test_cmake_bool_state(self) -> None:
        self.assertTrue(cmake_bool_state("ON"))
        self.assertFalse(cmake_bool_state("OFF"))
        self.assertIsNone(cmake_bool_state("MAYBE"))

    def test_read_cmake_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\nAMReX_GPU_BACKEND:STRING=NONE\n",
                encoding="utf-8",
            )
            entries = read_cmake_cache(build_dir, "test", "host-3d-release")
        self.assertEqual(entries["CMAKE_BUILD_TYPE"]["value"], "Release")
        self.assertEqual(entries["AMReX_GPU_BACKEND"]["value"], "NONE")


if __name__ == "__main__":
    unittest.main()
