from __future__ import annotations

import sys
from typing import Optional

import click
import typer

from quokka.cli import activation, bootstrap, build, clean, configure, doctor, format, list_cmd, lock, run, smoke, status, test, tidy
from quokka.cli.common import worktree_option


def create_app() -> typer.Typer:
    app = typer.Typer(
        name="quokka",
        add_completion=False,
        no_args_is_help=True,
        context_settings={"help_option_names": ["-h", "--help"]},
        pretty_exceptions_enable=False,
    )

    @app.callback()
    def callback(
        ctx: typer.Context,
        worktree: Optional[str] = worktree_option(),
    ) -> None:
        ctx.ensure_object(dict)
        ctx.obj["worktree"] = worktree

    build.register(app)
    configure.register(app)
    run.register(app)
    test.register(app)
    smoke.register(app)
    tidy.register(app)
    format.register(app)
    list_cmd.register(app)
    status.register(app)
    lock.register(app)
    clean.register(app)
    doctor.register(app)
    bootstrap.register(app)
    activation.register(app)
    return app


app = create_app()


def normalize_argv(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    worktree_args: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "-C":
            if index + 1 >= len(argv):
                normalized.append("--worktree")
                index += 1
                continue
            worktree_args = ["--worktree", argv[index + 1]]
            index += 2
            continue
        if arg == "--worktree":
            if index + 1 >= len(argv):
                normalized.append(arg)
                index += 1
                continue
            worktree_args = ["--worktree", argv[index + 1]]
            index += 2
            continue
        if arg.startswith("--worktree="):
            worktree_args = ["--worktree", arg.split("=", 1)[1]]
            index += 1
            continue
        normalized.append(arg)
        index += 1
    return worktree_args + normalized


def main() -> int:
    argv = normalize_argv(sys.argv[1:])
    try:
        app(args=argv, standalone_mode=False)
        return 0
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        return 1
