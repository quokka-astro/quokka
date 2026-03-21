# Repository Guidelines

## Project Structure & Module Organization
Core C++20 sources live in `src/`, with physics modules under `hydro/`, `radiation/`, `cooling/`, and `chemistry/`. Shared infrastructure such as `QuokkaSimulation.cpp` sits alongside module code, while scenario drivers compile from `src/problems/`. Runtime inputs (`*.in`) land in `inputs/`, docs in `docs/`, and helper utilities in `scripts/`. Generated builds and intermediates belong in `build/`; regression baselines and plotfiles stay under `tests/` and are tracked through `regression/quokka-tests.ini`.

## Build, Test, and Development Commands
Prefer the `quokka` CLI over raw `cmake` and `ctest` for routine local workflows. List available profiles with `quokka list profiles`, inspect problems with `quokka list problems --profile <profile>`, and build via `quokka build [target...] --profile <profile>`. Run a problem with `quokka run <problem> --input <input-file> --profile <profile>`, run tests with `quokka test [test-name] --profile <profile>` or `quokka test --ctest-regex <Pattern> --profile <profile>`, and run regression suites with `quokka regression [suite...] --profile <profile>`. Use `quokka status --profile <profile>` to inspect build and lock state, `quokka tidy [changed|previous|origin|dev] --profile <profile>` for clang-tidy, and `quokka format [changed|previous|origin|dev|all]` for formatting. Add `--build-if-needed` to `run`, `test`, or `regression` when those commands should build required targets first.

## Quokka CLI
Use the repository CLI for profile-aware local workflows.

Bootstrap the launcher once for interactive use:
`scripts/bash/install-quokka-bootstrap.sh`

Activate the current worktree in a shell:
`source scripts/bash/quokka-activate.sh`
`source scripts/bash/quokka-activate.sh host-default`

For non-interactive use, prefer explicit worktree selection:
`quokka -C /path/to/quokka list profiles`
`quokka -C /path/to/quokka build HydroWave --profile host-default`

Common commands:
`quokka list profiles`
`quokka list problems --profile <profile>`
`quokka build [target...] --profile <profile>`
`quokka run <problem> --input <input-file> --profile <profile>`
`quokka test [test-name] --profile <profile>`
`quokka test --ctest-regex <regex> --profile <profile>`
`quokka regression [suite...] --profile <profile>`
`quokka status --profile <profile>`
`quokka tidy [changed|previous|origin|dev] --profile <profile>`
`quokka format [changed|previous|origin|dev|all]`

Agents should verify `command -v quokka` before relying on the launcher. If it is unavailable, install it with `scripts/bash/install-quokka-bootstrap.sh` or invoke `python3 /path/to/quokka/scripts/python/quokka_cli.py -C /path/to/quokka ...` directly.

## Coding Style & Naming Conventions
Follow the repository `.clang-format` (LLVM-derived, 160-column width, tabs at eight spaces) and `.clang-tidy`. Keep headers as `.hpp` and implementations `.cpp`. Prefer PascalCase for classes and methods, camelCase with trailing underscore for data members, and wrap even single statements in braces. Favor trailing return types for non-`void` functions and mark variables `const` whenever possible.

## GPU Lambda Safety
Avoid capturing host pointers inside `AMREX_GPU_DEVICE` lambdas. Prefer device-safe value types like `amrex::GpuArray` (`geom.ProbLoArray()`, `geom.CellSizeArray()`, `geom.InvCellSizeArray()`). Do not pass `Geometry::ProbLo()/CellSize()` raw pointers into device lambdas; use the array forms instead. Never capture raw pointers from `GeometryData` inside GPU lambdas. Avoid accessing `GeometryData` directly; this is almost never required.

## Testing Guidelines
Add a unit test registered with CTest or extend `regression/quokka-tests.ini` with a matching `inputs/` file for new features. Preserve baseline outputs in `tests/` when regression scripts consume them, and document any GPU-only runs by recording the exact commands. Keep numerical changes reproducible with notes on key diagnostics.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects (e.g., `fix clang-tidy`). Group related changes only and rebase onto `development` before opening a PR. Provide PR descriptions covering problem, solution, and validation, cite the `ctest` or benchmark commands run, and link relevant issues. Flag interface or input changes for documentation follow-up and mention required plots or logs when numerics shift.
