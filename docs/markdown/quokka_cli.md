# Quokka CLI

The `quokka` command is the recommended interface for routine local work in a Quokka checkout. It provides a profile-aware workflow for building, running, testing, formatting, and inspecting a worktree.

For one-time build prerequisites and manual CMake instructions, see [Installation](installation.md). The CLI reads its profiles from `quokka.toml` at the repository root.

## Install the launcher

For normal interactive use, install the small `quokka` launcher once:

```bash
scripts/bash/install-quokka-bootstrap.sh
```

By default this installs `quokka` to `$HOME/.local/bin/quokka`. Make sure that directory is on your `PATH`.

## Activate a worktree

Activate the current checkout by sourcing the activation script:

```bash
source scripts/bash/quokka-activate.sh
source scripts/bash/quokka-activate.sh host-default
```

Activation records a default worktree and profile for the current shell, so you can run `quokka` from anywhere. It also adds a prompt prefix and defines `quokka_deactivate` to leave the environment.

If you omit the profile name, the CLI uses `policy.default_profile` from `quokka.toml`.

## Use the CLI without activation

For scripts or one-off commands, prefer explicit worktree selection with `-C`:

```bash
quokka -C /path/to/quokka status --profile host-default
quokka -C /path/to/quokka build HydroWave --profile host-default
```

If the launcher is not installed yet, you can invoke the worktree-local implementation directly:

```bash
python3 /path/to/quokka/scripts/python/quokka_cli.py -C /path/to/quokka status --profile host-default
```

Profile selection uses this order:

1. `--profile <name>`
2. the activated profile from `QUOKKA_PROFILE`
3. `policy.default_profile` from `quokka.toml`

## Quick start

This is a typical interactive workflow:

```bash
scripts/bash/install-quokka-bootstrap.sh
source scripts/bash/quokka-activate.sh host-default
quokka build
quokka list problems
quokka status
```

`quokka build` configures the selected profile if needed and builds the default target set. To build a specific problem instead, pass it explicitly:

```bash
quokka build <problem>
```

After the first successful configure/build, you can inspect the configured profile:

```bash
quokka list problems
quokka list tests
quokka list suites
```

## Common commands

### Build and run

Build one or more problems:

```bash
quokka build HydroWave --profile host-default
quokka build HydroWave OrszagTang --profile host-default
```

Run a problem executable with an input file:

```bash
quokka run <problem> --input <input-file> --profile host-default
```

By default, `run` validates that the executable is present and up to date. If you want it to build missing or stale targets first, add `--build-if-needed`:

```bash
quokka run <problem> --input <input-file> --build-if-needed --profile host-default
```

### Tests and regression suites

Run the full CTest suite for a configured profile:

```bash
quokka test --profile host-default
```

Run one named test or a regex selection:

```bash
quokka test <test-name> --profile host-default
quokka test --ctest-regex Hydro --profile host-default
```

Run one or more regression suites from `regression/quokka-tests.ini`:

```bash
quokka regression <suite> --profile host-default
quokka regression <suite1> <suite2> --profile host-default
```

As with `run`, both `test` and `regression` are validation-first by default. Add `--build-if-needed` when those commands should build required targets automatically.

### Status, formatting, and static analysis

Inspect the current profile, configure state, locks, and artifact freshness:

```bash
quokka status --profile host-default
```

Run `clang-tidy` on files selected relative to your Git history:

```bash
quokka tidy --profile host-default
quokka tidy previous --profile host-default
quokka tidy dev --fix --profile host-default
```

Run the repository `clang-format` hook:

```bash
quokka format
quokka format origin
quokka format all
```

`format` does not require a profile.

## Profiles

Profiles are defined in `quokka.toml`. Each profile selects a build directory, CMake generator, and CMake definitions such as dimensionality, MPI support, or GPU backend.

Use this command to discover available profiles:

```bash
quokka list profiles
```

If you need a different build configuration, add or edit a profile in `quokka.toml` and then build with `--profile <name>`.

## Machine-readable output

Most commands support `--json` for scripting:

```bash
quokka status --profile host-default --json
quokka list profiles --json
```

This prints a stable JSON result envelope with the command result or diagnostic error information.

## Getting help

Use the built-in help for the top-level CLI or an individual subcommand:

```bash
quokka --help
quokka build --help
quokka run --help
```

If `quokka` cannot resolve a worktree, either activate the checkout first or use `-C /path/to/quokka`.
