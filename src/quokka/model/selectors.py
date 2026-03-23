from __future__ import annotations

import re

from quokka.core.constants import FORMAT_SELECTORS, TIDY_SELECTORS
from quokka.core.errors import DiagnosticError
from quokka.core.types import TestRequest, TestSpec
from quokka.model.tests import discover_tests
from quokka.project.context import CliContext


def resolve_tidy_selector(selector: str, *, profile: str | None) -> str:
    if selector not in TIDY_SELECTORS:
        raise DiagnosticError(
            "TIDY_SELECTOR_INVALID",
            "Unsupported tidy selector '{}'".format(selector),
            command="tidy",
            profile=profile,
        )
    return selector


def resolve_format_selector(selector: str) -> str:
    if selector not in FORMAT_SELECTORS:
        raise DiagnosticError(
            "FORMAT_SELECTOR_INVALID",
            "Unsupported format selector '{}'".format(selector),
            command="format",
        )
    return selector


def ctest_selection(context: CliContext, request: TestRequest) -> list[TestSpec]:
    profile = context.require_profile("test")
    tests = discover_tests(profile.build_dir, "test", context.profile_name())
    if request.ctest_regex:
        pattern = re.compile(request.ctest_regex)
        matched = [test for test in tests if pattern.search(test.name)]
    elif request.test_name:
        matched = [test for test in tests if test.name == request.test_name]
    else:
        matched = tests
    if not matched:
        raise DiagnosticError(
            "UNKNOWN_RESOURCE",
            "No tests matched the requested selector.",
            command="test",
            profile=context.profile_name(),
            resource={"kind": "test", "name": request.test_name or request.ctest_regex or "*"},
        )
    return matched
