from __future__ import annotations

import dataclasses
import re
import shlex
from pathlib import Path

from quokka.core.constants import EXPECTATION_COMMENT_RE
from quokka.core.errors import DiagnosticError
from quokka.core.types import ProfileConfig, TestSpec
from quokka.model.targets import parse_source_problem_file, source_problem_cmakelists
from quokka.project.discovery import ctest_root_testfile_path


def parse_ctest_testfiles(build_dir: Path, command: str, profile: str | None) -> list[TestSpec]:
    root_testfile = ctest_root_testfile_path(build_dir)
    if not root_testfile.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' is not configured yet.".format(profile or "<none>"),
            command=command,
            profile=profile,
            details={"build_dir": str(build_dir)},
        )

    tests: list[TestSpec] = []
    pattern_add = re.compile(r"^add_test\((.*)\)$")
    pattern_props = re.compile(r"^set_tests_properties\((.*)\)$")
    pattern_subdirs = re.compile(r'^subdirs\("(.+)"\)$')

    pending = [root_testfile]
    visited = set()

    while pending:
        testfile = pending.pop(0).resolve()
        if testfile in visited:
            continue
        visited.add(testfile)
        by_name: dict[str, TestSpec] = {}
        try:
            lines = testfile.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DiagnosticError(
                "STATE_CORRUPT",
                "CTest metadata is unreadable.",
                command=command,
                profile=profile,
                details={"path": str(testfile)},
            ) from exc

        for raw_line in lines:
            line = raw_line.strip()
            match_subdirs = pattern_subdirs.match(line)
            if match_subdirs:
                child_testfile = (testfile.parent / match_subdirs.group(1) / "CTestTestfile.cmake").resolve()
                if child_testfile.exists():
                    pending.append(child_testfile)
                continue

            match_add = pattern_add.match(line)
            if match_add:
                parts = shlex.split(match_add.group(1))
                if len(parts) >= 2:
                    spec = TestSpec(
                        name=parts[0],
                        command=parts[1:],
                        working_directory=None,
                        source_path=testfile,
                    )
                    by_name[spec.name] = spec
                    tests.append(spec)
                continue

            match_props = pattern_props.match(line)
            if match_props:
                parts = shlex.split(match_props.group(1))
                if len(parts) < 3:
                    continue
                test_name = parts[0]
                if test_name not in by_name or "PROPERTIES" not in parts:
                    continue
                props = parts[parts.index("PROPERTIES") + 1 :]
                for index in range(0, len(props) - 1, 2):
                    if props[index] == "WORKING_DIRECTORY":
                        updated = dataclasses.replace(by_name[test_name], working_directory=Path(props[index + 1]).resolve())
                        by_name[test_name] = updated
                        for test_index, test in enumerate(tests):
                            if test.name == test_name and test.source_path == updated.source_path:
                                tests[test_index] = updated
                                break

    return tests


def discover_source_tests(worktree_root: Path, profile: ProfileConfig, command: str) -> list[TestSpec]:
    tests_by_name: dict[str, TestSpec] = {}
    for cmake_path in source_problem_cmakelists(worktree_root):
        _, file_tests = parse_source_problem_file(cmake_path, worktree_root, profile, command)
        for test in file_tests:
            tests_by_name.setdefault(test.name, test)
    return [tests_by_name[name] for name in sorted(tests_by_name)]


def discover_tests(build_dir: Path, command: str, profile: str | None) -> list[TestSpec]:
    return sorted(parse_ctest_testfiles(build_dir, command, profile), key=lambda spec: spec.name)


def problem_for_test(test: TestSpec) -> str:
    if not test.command:
        raise DiagnosticError(
            "TEST_MAPPING_UNSUPPORTED",
            "Test '{}' does not declare a runnable command.".format(test.name),
            command="test",
            resource={"kind": "test", "name": test.name},
        )
    executable = Path(test.command[0]).name
    if not executable:
        raise DiagnosticError(
            "TEST_MAPPING_UNSUPPORTED",
            "Test '{}' cannot be mapped to a single executable.".format(test.name),
            command="test",
            resource={"kind": "test", "name": test.name},
        )
    return executable


def source_file_for_test(context: "CliContext", test: TestSpec) -> Path | None:
    problem = problem_for_test(test)
    problem_dir = context.worktree_root / "src" / "problems" / problem
    preferred = problem_dir / "test{}.cpp".format(problem)
    if preferred.exists():
        return preferred

    if problem_dir.exists():
        candidates = sorted(problem_dir.glob("test*.cpp"))
        if len(candidates) == 1:
            return candidates[0]
    return None


def expectation_summary_for_test(context: "CliContext", test: TestSpec) -> dict[str, object] | None:
    source_path = source_file_for_test(context, test)
    if source_path is None:
        return None
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError:
        return None

    summaries: list[str] = []
    for line in text.splitlines():
        match = EXPECTATION_COMMENT_RE.match(line)
        if match:
            summaries.append(match.group(1).strip())
    if not summaries:
        return None
    return {
        "test": test.name,
        "summary": " ".join(summaries),
        "source": str(source_path),
    }


def test_map_by_name(context: "CliContext", command: str) -> dict[str, TestSpec]:
    profile = context.require_profile(command)
    return {test.name: test for test in discover_tests(profile.build_dir, command, context.profile_name())}
