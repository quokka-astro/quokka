from __future__ import annotations

from quokka.core.errors import DiagnosticError
from quokka.core.result import CommandResult
from quokka.core.types import ListRequest
from quokka.model.targets import discover_problems, discover_source_problems
from quokka.model.tests import discover_source_tests, discover_tests
from quokka.project.context import CliContext
from quokka.project.presets import list_profiles
from quokka.project.state import is_build_configured


def run_list(context: CliContext, request: ListRequest) -> CommandResult:
    resource = None
    if request.list_kind == "profiles":
        profile_names = list_profiles(context.config)
        data = {"profiles": profile_names}
        text = "\n".join(profile_names)
        return CommandResult("list", None, {"kind": "profiles", "name": "*"}, data, text)

    profile = context.require_profile("list")
    configured = is_build_configured(profile.build_dir)
    if request.list_kind == "problems":
        discovery = "build" if configured else "source"
        problems = discover_problems(profile.build_dir, "list", context.profile_name()) if configured else discover_source_problems(context.worktree_root, profile, "list")
        data = {"problems": problems, "discovery": discovery}
        text = "\n".join(problems)
        resource = {"kind": "problem", "name": "*"}
    elif request.list_kind == "tests":
        discovery = "build" if configured else "source"
        tests = [test.name for test in (discover_tests(profile.build_dir, "list", context.profile_name()) if configured else discover_source_tests(context.worktree_root, profile, "list"))]
        data = {"tests": tests, "discovery": discovery}
        text = "\n".join(tests)
        resource = {"kind": "test", "name": "*"}
    else:
        raise DiagnosticError("USAGE_ERROR", "Unsupported list kind '{}'.".format(request.list_kind), command="list")
    return CommandResult("list", context.profile_name(), resource, data, text)
