import argparse
import dataclasses
import pathlib
import re

WORKFLOWS_DIRECTORY = pathlib.Path(".github/workflows")

##
## === RUNNER OS
##


@dataclasses.dataclass(frozen=True)
class RunnerOS:
    name: str
    label_pattern: re.Pattern[str]


@dataclasses.dataclass(frozen=True)
class OSUpdate:
    runner_os: RunnerOS
    target_version: str


UBUNTU_RUNNER_OS = RunnerOS(
    name="ubuntu",
    label_pattern=re.compile(r"runs-on:\s*ubuntu-(?P<version>\d+\.\d+)\b"),
)
MACOS_RUNNER_OS = RunnerOS(
    name="macos",
    label_pattern=re.compile(r"runs-on:\s*macos-(?P<version>\d+)\b"),
)

##
## === LABEL UPDATES
##


def _parse_version(
    version: str,
) -> tuple[int, ...]:
    return tuple(int(_version_part) for _version_part in version.split("."))


def _update_line(
    *,
    line: str,
    os_update: OSUpdate,
) -> str:
    label_match = os_update.runner_os.label_pattern.search(line)
    if label_match is None:
        return line
    current_version = label_match.group("version")
    parsed_current_version = _parse_version(current_version)
    parsed_target_version = _parse_version(os_update.target_version)
    if parsed_current_version >= parsed_target_version:
        return line
    return line[:label_match.start("version")] + os_update.target_version + line[label_match.end("version"):]


def update_workflow_lines(
    workflow_lines: tuple[str, ...],
    *,
    os_updates: tuple[OSUpdate, ...],
) -> tuple[str, ...]:
    updated_lines = workflow_lines
    for os_update in os_updates:
        updated_lines = tuple(_update_line(
            line=_line,
            os_update=os_update,
        ) for _line in updated_lines)
    return updated_lines


def update_workflow_file(
    workflow_path: pathlib.Path,
    *,
    os_updates: tuple[OSUpdate, ...],
) -> None:
    workflow_lines = tuple(workflow_path.read_text().splitlines(keepends=True))
    updated_lines = update_workflow_lines(workflow_lines, os_updates=os_updates)
    if updated_lines == workflow_lines:
        return
    workflow_path.write_text("".join(updated_lines))
    for current_line, updated_line in zip(workflow_lines, updated_lines):
        if current_line != updated_line:
            print(f"{workflow_path.as_posix()}: {current_line.strip()} -> {updated_line.strip()}")


##
## === PROGRAM MAIN
##


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ubuntu-version",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--macos-version",
        type=str,
        required=True,
    )
    parsed_args = parser.parse_args()
    os_updates = (
        OSUpdate(
            runner_os=UBUNTU_RUNNER_OS,
            target_version=parsed_args.ubuntu_version,
        ),
        OSUpdate(
            runner_os=MACOS_RUNNER_OS,
            target_version=parsed_args.macos_version,
        ),
    )
    for workflow_path in sorted(WORKFLOWS_DIRECTORY.glob("*.yml")):
        update_workflow_file(workflow_path, os_updates=os_updates)


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()
