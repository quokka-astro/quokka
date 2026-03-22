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
quokka doctor --profile host-default
```

`quokka build` configures the selected profile if needed and builds the default target set. To build a specific problem instead, pass it explicitly:

```bash
quokka build <problem>
```

After the first successful configure/build, you can inspect the configured profile:

```bash
quokka list problems
quokka list tests
```

## Fast local validation

For first-time local verification, start with a small smoke test rather than a long hydro or radiation problem.

- Profile choice matters. `host-default` is the default profile, but it is a `Debug` configuration intended for debug-oriented validation. `host-3d` is a faster `Release` profile and is usually the better first choice for local smoke tests.
- `quokka smoke` is the shortest first-run path. It checks the runtime and selected profile, configures the build tree if needed, and runs a small recommended test. By default it uses `ODEIntegration`.

```bash
quokka smoke --profile host-3d
```

- Check which profiles are available before running anything expensive:

```bash
quokka list profiles
quokka status --profile host-3d
quokka doctor --profile host-3d
```

- A good first smoke test is `ODEIntegration`, which builds quickly and has an explicit numerical tolerance:

```bash
quokka test ODEIntegration --profile host-3d --build-if-needed
```

- If you want to run the executable directly with its input file, use:

```bash
quokka run ODEIntegration --input inputs/ODEIntegration.toml --profile host-3d --build-if-needed
```

- Use `host-default` when debug instrumentation or the repository default configuration matters more than speed:

```bash
quokka test ODEIntegration --profile host-default --build-if-needed
```

- If a build or test fails unexpectedly, inspect the current toolchain, locks, and profile state before retrying:

```bash
quokka doctor --profile host-3d
quokka doctor runtime --profile host-3d
quokka doctor profile --profile host-3d
quokka doctor locking --profile host-3d
```

- Do not treat timings from one profile as representative of another. The biggest runtime drivers are build type (`Debug` vs. `Release`), MPI enablement, and CPU vs. GPU backend.
- `HydroWave`, `Advection`, and similar evolution problems are useful tests, but they are not ideal first smoke tests because they advance to a fixed physical time and can take much longer than `ODEIntegration`.
- The GPU/CI regression harness is not exposed through the `quokka` CLI in this repository. Use the local `build`, `run`, and `test` workflow for routine validation.
- If you need live CTest progress and the full stdout/stderr stream while diagnosing a failing test, rerun with `--stream`:

```bash
quokka test ODEIntegration --profile host-3d --stream
```

- If you want shorter live progress while still keeping the full configure/build/test output, use `--compact-stream`. The CLI prints a concise progress summary and writes the complete log to `QUOKKA_RUNTIME_DIR/runs/`:

```bash
quokka smoke --profile host-3d --compact-stream
quokka test ODEIntegration --profile host-3d --build-if-needed --compact-stream
```

- If you still need more direct test control, fall back to raw CTest in the selected build directory after the first configure/build:

```bash
cd build/host-3d
ctest --output-on-failure -R ODEIntegration
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

### Tests

Run the full CTest suite for a configured profile:

```bash
quokka test --profile host-default
```

Run one named test or a regex selection:

```bash
quokka test <test-name> --profile host-default
quokka test --ctest-regex Hydro --profile host-default
```

When diagnosing a failure, add `--stream` to show live progress and test stdout/stderr:

```bash
quokka test <test-name> --profile host-default --stream
quokka test --ctest-regex Hydro --profile host-default --stream
```

For a shorter console summary with a full log file on disk, use `--compact-stream` instead:

```bash
quokka test <test-name> --profile host-default --compact-stream
```

As with `run`, `test` is validation-first by default. Add `--build-if-needed` when the command should build required targets automatically.

### Diagnostics, status, formatting, and static analysis

Inspect the runtime toolchain, Python stack, lock state, and profile drift:

```bash
quokka doctor --profile host-default
quokka doctor runtime --profile host-default
quokka doctor profile --profile host-default
quokka doctor locking --profile host-default
```

Inspect the current profile, configure state, locks, and artifact freshness:

```bash
quokka status --profile host-default
```

In `status`, `not_built` means the problem is known for the configured profile but has not been compiled yet. It is distinct from stale or broken build metadata.

Run `clang-tidy` on files selected relative to your Git history:

```bash
quokka tidy --profile host-default
quokka tidy previous --profile host-default
quokka tidy dev --fix --profile host-default
```

Selectors:
- `changed` (default): files modified in the working tree relative to `HEAD`
- `previous`: files modified in the previous commit
- `origin`: files different from `origin/<current-branch>`
- `dev`: files different from the local `development` branch

Prerequisites:
- the selected profile must already be configured
- `compile_commands.json` must exist in the selected build directory

Run the repository `clang-format` hook:

```bash
quokka format
quokka format origin
quokka format all
```

Selectors:
- `changed` (default): files modified in the working tree relative to `HEAD`
- `previous`: files modified in the previous commit
- `origin`: files different from `origin/<current-branch>`
- `dev`: files different from the local `development` branch
- `all`: all files covered by the `clang-format` pre-commit hook

Prerequisites:
- `pre-commit` must be installed and available on `PATH`
- `format` uses the repository hook definition from `.pre-commit-config.yaml`

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
quokka test --help
quokka doctor --help
quokka smoke --help
```

If `quokka` cannot resolve a worktree, either activate the checkout first or use `-C /path/to/quokka`.
