from __future__ import annotations

import unittest
from pathlib import Path

from quokka.project.config import load_repo_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_load_repo_config(self) -> None:
        config = load_repo_config(ROOT)
        self.assertEqual(config.default_profile, "host-3d-release")
        self.assertIn("host-3d-release", config.profiles)
        self.assertTrue(config.profiles["host-3d-release"].build_dir.name.endswith("release"))


if __name__ == "__main__":
    unittest.main()
