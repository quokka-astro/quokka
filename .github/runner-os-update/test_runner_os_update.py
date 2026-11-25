import unittest

import runner_os_update

OS_UPDATES = (
    runner_os_update.OSUpdate(
        runner_os=runner_os_update.UBUNTU_RUNNER_OS,
        target_version="26.04",
    ),
    runner_os_update.OSUpdate(
        runner_os=runner_os_update.MACOS_RUNNER_OS,
        target_version="26",
    ),
)


def update_line(
    line: str,
) -> str:
    updated_lines = runner_os_update.update_workflow_lines((line, ), os_updates=OS_UPDATES)
    return updated_lines[0]


class TestRunnerOSUpdate_Ubuntu(unittest.TestCase):

    def test_older_label_is_raised(
        self,
    ):
        self.assertEqual(update_line("    runs-on: ubuntu-24.04\n"), "    runs-on: ubuntu-26.04\n")

    def test_arm_suffix_is_kept(
        self,
    ):
        self.assertEqual(update_line("    runs-on: ubuntu-24.04-arm\n"), "    runs-on: ubuntu-26.04-arm\n")

    def test_newer_label_is_never_lowered(
        self,
    ):
        self.assertEqual(update_line("    runs-on: ubuntu-28.04\n"), "    runs-on: ubuntu-28.04\n")

    def test_current_label_is_left(
        self,
    ):
        self.assertEqual(update_line("    runs-on: ubuntu-26.04\n"), "    runs-on: ubuntu-26.04\n")

    def test_latest_label_is_left(
        self,
    ):
        self.assertEqual(update_line("    runs-on: ubuntu-latest\n"), "    runs-on: ubuntu-latest\n")

    def test_label_outside_runs_on_is_left(
        self,
    ):
        self.assertEqual(update_line("    image: ubuntu-24.04\n"), "    image: ubuntu-24.04\n")


class TestRunnerOSUpdate_Macos(unittest.TestCase):

    def test_older_label_is_raised(
        self,
    ):
        self.assertEqual(update_line("    runs-on: macos-15\n"), "    runs-on: macos-26\n")

    def test_latest_label_is_left(
        self,
    ):
        self.assertEqual(update_line("    runs-on: macos-latest\n"), "    runs-on: macos-latest\n")


if __name__ == "__main__":
    unittest.main()
