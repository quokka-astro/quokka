from __future__ import annotations

import dataclasses
import os
import re
import shlex
from pathlib import Path

from quokka.core.errors import DiagnosticError
from quokka.core.types import ProfileConfig, TestSpec
from quokka.project.discovery import ctest_root_testfile_path


def strip_cmake_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        chars: list[str] = []
        in_quote = False
        escaped = False
        for ch in raw_line:
            if ch == '"' and not escaped:
                in_quote = not in_quote
            if ch == "#" and not in_quote:
                break
            chars.append(ch)
            if ch == "\\" and not escaped:
                escaped = True
            else:
                escaped = False
        cleaned_lines.append("".join(chars))
    return "\n".join(cleaned_lines)


def iter_cmake_invocations(text: str):
    source = strip_cmake_comments(text)
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    index = 0
    while index < len(source):
        match = pattern.search(source, index)
        if match is None:
            break
        name = match.group(0).lower()
        cursor = match.end()
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor >= len(source) or source[cursor] != "(":
            index = match.end()
            continue

        depth = 1
        cursor += 1
        body_start = cursor
        in_quote = False
        escaped = False
        while cursor < len(source) and depth > 0:
            ch = source[cursor]
            if ch == '"' and not escaped:
                in_quote = not in_quote
            elif not in_quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            if ch == "\\" and in_quote and not escaped:
                escaped = True
            else:
                escaped = False
            cursor += 1

        if depth != 0:
            break

        yield name, source[body_start:cursor]
        index = cursor + 1


def split_cmake_arguments(body: str) -> list[str]:
    compact = body.replace("\n", " ")
    try:
        return shlex.split(compact, comments=False, posix=True)
    except ValueError:
        return compact.split()


def expand_cmake_token(token: str, variables: dict[str, str]) -> str:
    return re.sub(r"\$\{([^}]+)\}", lambda match: variables.get(match.group(1), match.group(0)), token)


def normalize_cmake_boolean(value: str) -> bool | None:
    upper = value.upper()
    if upper in {"1", "ON", "TRUE", "YES", "Y"}:
        return True
    if upper in {"0", "OFF", "FALSE", "NO", "N", "IGNORE", "NOTFOUND", ""}:
        return False
    return None


def evaluate_source_condition(tokens: list[str], profile: ProfileConfig, variables: dict[str, str]) -> bool | None:
    expanded = [expand_cmake_token(token, variables) for token in tokens]
    if not expanded:
        return None

    if len(expanded) == 1:
        token = expanded[0]
        value = variables.get(token, profile.defines.get(token, token))
        return normalize_cmake_boolean(str(value))

    lhs_token = expanded[0]
    operator = expanded[1].upper()
    rhs_token = expanded[2] if len(expanded) >= 3 else ""
    lhs = str(variables.get(lhs_token, profile.defines.get(lhs_token, lhs_token)))
    rhs = str(variables.get(rhs_token, profile.defines.get(rhs_token, rhs_token)))

    if operator == "MATCHES":
        try:
            return re.search(rhs, lhs) is not None
        except re.error:
            return None

    if operator in {"EQUAL", "GREATER_EQUAL"}:
        try:
            lhs_num = int(lhs)
            rhs_num = int(rhs)
        except ValueError:
            if operator == "EQUAL":
                return lhs == rhs
            return None
        if operator == "EQUAL":
            return lhs_num == rhs_num
        return lhs_num >= rhs_num

    if operator == "STREQUAL":
        return lhs == rhs

    return None


def source_problem_cmakelists(worktree_root: Path) -> list[Path]:
    problems_root = worktree_root / "src" / "problems"
    if not problems_root.exists():
        raise DiagnosticError(
            "STATE_CORRUPT",
            "Problem source directory is missing.",
            command="list",
            details={"path": str(problems_root)},
        )
    return sorted(path for path in problems_root.glob("*/CMakeLists.txt") if path.is_file())


def resolve_source_working_directory(raw_value: str, worktree_root: Path, source_dir: Path) -> Path:
    resolved = raw_value.replace("${CMAKE_SOURCE_DIR}", str(worktree_root)).replace("${CMAKE_CURRENT_SOURCE_DIR}", str(source_dir))
    path = Path(resolved)
    if not path.is_absolute():
        path = (source_dir / path).resolve()
    return path.resolve()


def source_testspec_from_add_test(tokens: list[str], variables: dict[str, str], cmake_path: Path, worktree_root: Path) -> TestSpec | None:
    expanded = [expand_cmake_token(token, variables) for token in tokens]
    if not expanded:
        return None

    if "NAME" not in expanded:
        if len(expanded) < 2:
            return None
        return TestSpec(name=expanded[0], command=list(expanded[1:]), working_directory=None, source_path=cmake_path)

    name_index = expanded.index("NAME")
    if name_index + 1 >= len(expanded):
        return None
    test_name = expanded[name_index + 1]

    command_tokens: list[str] = []
    if "COMMAND" in expanded:
        command_index = expanded.index("COMMAND")
        command_end = len(expanded)
        for keyword in ("WORKING_DIRECTORY", "CONFIGURATIONS", "COMMAND_EXPAND_LISTS"):
            if keyword in expanded[command_index + 1 :]:
                command_end = min(command_end, expanded.index(keyword, command_index + 1))
        command_tokens = list(expanded[command_index + 1 : command_end])

    working_directory: Path | None = None
    if "WORKING_DIRECTORY" in expanded:
        wd_index = expanded.index("WORKING_DIRECTORY")
        if wd_index + 1 < len(expanded):
            working_directory = resolve_source_working_directory(expanded[wd_index + 1], worktree_root, cmake_path.parent)

    return TestSpec(name=test_name, command=command_tokens, working_directory=working_directory, source_path=cmake_path)


def source_testspec_from_quokka_add_problem(tokens: list[str], variables: dict[str, str], cmake_path: Path, worktree_root: Path) -> TestSpec | None:
    kwargs: dict[str, str] = {}
    recognized = {"JOB_NAME", "INPUT_FILE", "ADD_TEST", "TEST_PARAMS", "PRIORITY"}
    index = 0
    while index < len(tokens):
        key = tokens[index].upper()
        if key in recognized and index + 1 < len(tokens):
            kwargs[key] = expand_cmake_token(tokens[index + 1], variables)
            index += 2
            continue
        index += 1

    job_name = kwargs.get("JOB_NAME")
    if not job_name:
        return None
    if kwargs.get("ADD_TEST", "ON").upper() == "OFF":
        return None

    input_file = kwargs.get("INPUT_FILE", "{}.toml".format(job_name))
    return TestSpec(
        name=job_name,
        command=[job_name, "../inputs/{}".format(input_file), "${QuokkaTestParams}"],
        working_directory=(worktree_root / "tests").resolve(),
        source_path=cmake_path,
    )


def parse_source_problem_file(cmake_path: Path, worktree_root: Path, profile: ProfileConfig, command: str) -> tuple[set[str], list[TestSpec]]:
    try:
        text = cmake_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError(
            "STATE_CORRUPT",
            "Source metadata is unreadable.",
            command=command,
            profile=profile.name,
            details={"path": str(cmake_path)},
        ) from exc

    problems: set[str] = set()
    tests: list[TestSpec] = []
    variables: dict[str, str] = {}
    active_stack = [True]
    matched_stack: list[bool] = []

    for invocation, body in iter_cmake_invocations(text):
        tokens = split_cmake_arguments(body)

        if invocation == "if":
            condition = evaluate_source_condition(tokens, profile, variables)
            parent_active = active_stack[-1]
            branch_active = parent_active and condition is not False
            active_stack.append(branch_active)
            matched_stack.append(condition is not False)
            continue

        if invocation == "elseif":
            if len(active_stack) == 1 or not matched_stack:
                continue
            parent_active = active_stack[-2]
            already_matched = matched_stack[-1]
            condition = evaluate_source_condition(tokens, profile, variables)
            branch_matches = condition is not False
            active_stack[-1] = parent_active and (not already_matched) and branch_matches
            matched_stack[-1] = already_matched or branch_matches
            continue

        if invocation == "else":
            if len(active_stack) == 1 or not matched_stack:
                continue
            parent_active = active_stack[-2]
            active_stack[-1] = parent_active and (not matched_stack[-1])
            matched_stack[-1] = True
            continue

        if invocation == "endif":
            if len(active_stack) > 1:
                active_stack.pop()
            if matched_stack:
                matched_stack.pop()
            continue

        if not active_stack[-1]:
            continue

        if invocation == "set" and len(tokens) >= 2:
            variables[tokens[0]] = expand_cmake_token(tokens[1], variables)
            continue

        if invocation == "quokka_add_problem":
            spec = source_testspec_from_quokka_add_problem(tokens, variables, cmake_path, worktree_root)
            job_name = None
            for index, token in enumerate(tokens[:-1]):
                if token.upper() == "JOB_NAME":
                    job_name = expand_cmake_token(tokens[index + 1], variables)
                    break
            if job_name:
                problems.add(job_name)
            if spec is not None:
                tests.append(spec)
            continue

        if invocation == "add_executable" and tokens:
            target = expand_cmake_token(tokens[0], variables)
            if target:
                problems.add(target)
            continue

        if invocation == "add_test":
            spec = source_testspec_from_add_test(tokens, variables, cmake_path, worktree_root)
            if spec is not None:
                tests.append(spec)

    return problems, tests


def discover_source_problems(worktree_root: Path, profile: ProfileConfig, command: str) -> list[str]:
    problems: set[str] = set()
    for cmake_path in source_problem_cmakelists(worktree_root):
        problems.add(cmake_path.parent.name)
        file_problems, _ = parse_source_problem_file(cmake_path, worktree_root, profile, command)
        problems.update(file_problems)
    return sorted(problems)


def discover_problems(build_dir: Path, command: str, profile: str | None) -> list[str]:
    root_testfile = ctest_root_testfile_path(build_dir)
    if not root_testfile.exists():
        raise DiagnosticError(
            "PROFILE_UNCONFIGURED",
            "Profile '{}' is not configured yet.".format(profile or "<none>"),
            command=command,
            profile=profile,
            details={"build_dir": str(build_dir)},
        )

    from quokka.model.tests import parse_ctest_testfiles

    problems = set()
    pattern_subdirs = re.compile(r'^subdirs\("(.+)"\)$')
    problems_index = build_dir / "src" / "problems" / "CTestTestfile.cmake"
    if problems_index.exists():
        try:
            lines = problems_index.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DiagnosticError(
                "STATE_CORRUPT",
                "CTest metadata is unreadable.",
                command=command,
                profile=profile,
                details={"path": str(problems_index)},
            ) from exc

        for raw_line in lines:
            line = raw_line.strip()
            match_subdirs = pattern_subdirs.match(line)
            if match_subdirs:
                problem_name = Path(match_subdirs.group(1)).name
                if problem_name:
                    problems.add(problem_name)

    tests = parse_ctest_testfiles(build_dir, command, profile)
    for test in tests:
        source_parent = test.source_path.resolve().parent
        if source_parent.parent == problems_index.parent:
            problems.add(source_parent.name)
            continue

        for argument in test.command:
            candidate = Path(argument)
            if candidate.parent == problems_index.parent / candidate.stem:
                problems.add(candidate.stem)
                break

    from quokka.project.state import artifact_receipts_dir

    receipts = artifact_receipts_dir(build_dir)
    if receipts.exists():
        for receipt in receipts.glob("*.json"):
            problems.add(receipt.stem)

    problems_root = build_dir / "src" / "problems"
    if problems_root.exists():
        for candidate in problems_root.glob("*/*"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                problems.add(candidate.name)

    return sorted(problems)
