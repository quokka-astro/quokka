from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTest(unittest.TestCase):
    def run_cli(self, *args: str, use_script: bool = False, use_launcher: bool = False) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        env["PYTHONPYCACHEPREFIX"] = "/tmp/pycache"
        if use_script:
            command = [sys.executable, str(ROOT / "scripts" / "python" / "quokka_cli.py"), *args]
        elif use_launcher:
            command = ["bash", str(ROOT / "scripts" / "bash" / "_quokka-launcher.sh"), *args]
        else:
            command = [sys.executable, "-m", "quokka", *args]
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_module_entrypoint_list_profiles_json(self) -> None:
        proc = self.run_cli("-C", str(ROOT), "list", "profiles", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("host-3d-release", payload["data"]["profiles"])

    def test_legacy_script_shim_list_profiles_json(self) -> None:
        proc = self.run_cli("-C", str(ROOT), "list", "profiles", "--json", use_script=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("host-3d-release", payload["data"]["profiles"])

    def test_launcher_list_profiles_json(self) -> None:
        proc = self.run_cli("-C", str(ROOT), "list", "profiles", "--json", use_launcher=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("host-3d-release", payload["data"]["profiles"])

    def test_module_entrypoint_activate_env(self) -> None:
        proc = self.run_cli("-C", str(ROOT), "_activate-env", "--profile", "host-3d-release")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("export QUOKKA_PROFILE=host-3d-release", proc.stdout)

    def test_module_entrypoint_doctor_runtime_json(self) -> None:
        proc = self.run_cli("-C", str(ROOT), "doctor", "runtime", "--profile", "host-3d-release", "--json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["command"], "doctor")
        self.assertIn("runtime", payload["data"])


if __name__ == "__main__":
    unittest.main()
