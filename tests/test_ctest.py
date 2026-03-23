from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quokka.tools.ctest import extract_metric_lines, parse_ctest_lasttest_output


class CTestTest(unittest.TestCase):
    def test_parse_lasttest_output(self) -> None:
        log_text = "\n".join(
            [
                "Test: ODEIntegration",
                "Output:",
                "final temperature = 1.0e4",
                "<end of output>",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "LastTest.log"
            log_path.write_text(log_text, encoding="utf-8")
            parsed = parse_ctest_lasttest_output(log_path)
        self.assertEqual(parsed["ODEIntegration"], ["final temperature = 1.0e4"])

    def test_extract_metric_lines(self) -> None:
        metrics = extract_metric_lines(["initial noise", "L1 error = 1.2e-3", "elapsed time 3s"])
        self.assertEqual(metrics, ["L1 error = 1.2e-3"])


if __name__ == "__main__":
    unittest.main()
