# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++20 codebase.

## Build & Test Commands
- Prefer the `quokka` CLI over raw `cmake` and `ctest` for routine local workflows.
- Bootstrap the launcher once with `scripts/bash/install-quokka-bootstrap.sh`.
- Activate a shell with `source scripts/bash/quokka-activate.sh` or `source scripts/bash/quokka-activate.sh host-default`.
- For non-interactive use, prefer `quokka -C /path/to/quokka ...`.
- List profiles with `quokka list profiles`.
- Build problems with `quokka build [target...] --profile <profile>`.
- Run a problem with `quokka run <problem> --input <input-file> --profile <profile>`.
- Run tests with `quokka test [test-name] --profile <profile>` or `quokka test --ctest-regex <Pattern> --profile <profile>`.
- Run regression suites with `quokka regression [suite...] --profile <profile>`.
- Use `quokka status --profile <profile>` to inspect build and lock state.
- Use `quokka tidy [changed|previous|origin|dev] --profile <profile>` for clang-tidy.
- Use `quokka format [changed|previous|origin|dev|all]` for formatting.
- Add `--build-if-needed` to `run`, `test`, or `regression` when those commands should build required targets first.
- Build configuration, including dimensionality and GPU backend, is selected through profiles defined in `quokka.toml`.
- Runtime inputs are located in `inputs/` (primarily `.toml` files).
- Agents should verify `command -v quokka` before relying on the launcher. If it is unavailable, install it or invoke `python3 /path/to/quokka/scripts/python/quokka_cli.py -C /path/to/quokka ...` directly.

## Profile Selection & Runtime Expectations
- Profile choice has a large effect on local runtime. Check `quokka.toml` or `quokka list profiles` before drawing conclusions from slow builds or tests.
- `host-default` is the default profile and is intended for debug-oriented local validation. It uses `build/`, `CMAKE_BUILD_TYPE=Debug`, `AMReX_MPI=ON`, and `AMReX_GPU_BACKEND=NONE`. Expect slower compile and test times than a release build.
- `host-3d` is the faster local iteration profile in this checkout. It uses `build/host-3d`, `CMAKE_BUILD_TYPE=Release`, and `AMReX_MPI=OFF`. Prefer it for quick smoke tests unless MPI-specific behavior or debug instrumentation matters for the task.
- The biggest runtime drivers are build type (`Debug` vs. `Release`), MPI enablement, and CPU vs. GPU backend. Do not treat timings from one profile as representative of another.
- Before reporting that a test is unexpectedly slow, confirm which profile you used and mention it explicitly in the summary.
- For first-time verification, prefer a quick smoke-test path before running longer hydro or radiation problems. A good starting point is `quokka test ODEIntegration --profile host-3d --build-if-needed` when that profile is available.
- Some problem tests are inherently longer because the executable advances to a fixed physical time in the problem code. If a run seems slow, inspect the problem source and input file before assuming the wrapper is the bottleneck.
- The regression suites listed by `quokka regression` are primarily GPU/CI-oriented in this repository. Treat them as specialized validation, not as the default local smoke-test path.

## Architecture Overview
- **Main entry**: `src/main.cpp` calls `problem_main()` defined in problem-specific files
- **Core simulation**: `QuokkaSimulation` template class inherits from `AMRSimulation`
- **Physics modules**: Located in `src/hydro/`, `src/radiation/`, `src/cooling/`, `src/chemistry/`
- **Hyperbolic systems**: `HyperbolicSystem` template handles conservation laws and slope limiters
- **Problem definitions**: Each problem in `src/problems/` has `.cpp/.hpp` files and CMake target
- **I/O and diagnostics**: `src/io/` contains output handling (plotfiles, checkpoints, openPMD)
- **Math utilities**: `src/math/` has interpolation, quadrature, root finding, ODE integration
- **Particles**: `src/particles/` handles stellar particles with accretion, creation, destruction

## Problem Structure
- Each problem directory contains:
  - `test_*.cpp`: Implementation with initial conditions and problem-specific physics
  - `test_*.hpp`: Header with template specializations (removed in recent commits)
  - `CMakeLists.txt`: Defines executable target
- Problems use template specialization pattern for `QuokkaSimulation<ProblemName>`
- Input files (`.in`) in `inputs/` configure geometry, AMR, physics parameters
- Problems should ONLY contain `.cpp` files (no `.hpp` files per recent policy)

## Key Dependencies
- **AMReX**: Underlying AMR framework (external submodule)
- **Microphysics**: Nuclear reaction networks (external submodule)
- **fmt, yaml-cpp**: Formatting and configuration parsing
- **HDF5**: I/O backend
- **OpenPMD-api**: Optional for large-scale output
- **Python**: Optional for analysis tools

## Code Style Guidelines
- Use `.clang-format` from `src/` directory for formatting (LLVM-based style)
- 160 character line limit, 8-space indentation with tabs
- Classes use PascalCase (e.g., `QuokkaSimulation`)
- Member variables use camelCase with trailing underscore (e.g., `radiationCflNumber_`)
- Member functions use PascalCase (e.g., `ReadCheckpointFile`)
- Always use curly braces for single statement blocks
- Always use a trailing return type for functions that do not return `void`
- ALWAYS declare variables `const` when they are never modified after initialization.
- Document APIs using Doxygen style comments
- PRs should be focused on a single change and target the `development` branch
- Static analysis with clang-tidy available for code quality checks
- Comprehensive clang-tidy configuration in `src/.clang-tidy` with extensive checks enabled
