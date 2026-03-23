from __future__ import annotations

import unittest

from quokka.workflows.common import summarize_runtime_output


class WorkflowTestTest(unittest.TestCase):
    def test_summarize_runtime_output_prefers_metrics(self) -> None:
        lines = summarize_runtime_output("L1 error = 2.5e-4\n", "")
        self.assertEqual(lines, ["L1 error = 2.5e-4"])


if __name__ == "__main__":
    unittest.main()
