from __future__ import annotations

import unittest
from pathlib import Path

from quokka.project.root import find_worktree_from_cwd


ROOT = Path(__file__).resolve().parents[1]


class RootTest(unittest.TestCase):
    def test_find_worktree_from_nested_path(self) -> None:
        nested = ROOT / "src" / "quokka" / "cli"
        self.assertEqual(find_worktree_from_cwd(nested), ROOT)


if __name__ == "__main__":
    unittest.main()
