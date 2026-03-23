from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from quokka.model.files import select_format_files, select_tidy_files


class FakeContext:
    worktree_root = Path("/tmp/worktree")

    def profile_name(self) -> str:
        return "host-3d-release"


class ModelFilesTest(unittest.TestCase):
    def test_select_format_files_all_uses_tracked_files(self) -> None:
        context = FakeContext()
        with patch(
            "quokka.model.files.git_tracked_files",
            return_value=["src/HydroWave.cpp", "README.md", "src/HydroWave.H"],
        ) as tracked_files:
            selection = select_format_files(context, "all")

        tracked_files.assert_called_once_with(context.worktree_root, "format")
        self.assertTrue(selection.all_files)
        self.assertEqual(selection.files, ["src/HydroWave.cpp", "src/HydroWave.H"])
        self.assertEqual(selection.skipped_files, ["README.md"])

    def test_select_tidy_files_filters_non_cpp_paths(self) -> None:
        context = FakeContext()
        with patch(
            "quokka.model.files.git_changed_files",
            return_value=["src/HydroWave.cpp", "docs/notes.txt", "src/HydroWave.hpp"],
        ) as changed_files:
            selection = select_tidy_files(context, "changed")

        changed_files.assert_called_once_with(context.worktree_root, "changed", "tidy", context.profile_name())
        self.assertFalse(selection.all_files)
        self.assertEqual(selection.files, ["src/HydroWave.cpp", "src/HydroWave.hpp"])
        self.assertEqual(selection.skipped_files, ["docs/notes.txt"])


if __name__ == "__main__":
    unittest.main()
