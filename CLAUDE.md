# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++20 codebase.

## Build & Test Commands

The `scripts/bash/quokka` script is the recommended way to configure, build, and run tests. Make sure it exists in your `PATH`; if not, install it once by copying it to a directory on your `PATH` (e.g. `~/.local/bin/`). All commands accept `--root <path>` to specify the repo root when not running from it.

The script optionally sources an environment file via `--source <file>` for commands that need the build/test environment (`config`, `build`, `buildrun`, `run`, `target`). If `--source` is omitted, no environment file is sourced.

- **Configure**: `quokka config [-d <preset>] [--delete] [--source <file>] [-D<k>=<v> ...]` — runs CMake with the selected preset (default `1d`).
- **Build one or more problems**: `quokka build [-d <preset>] <problem> [<problem> ...] [-j <N>] [--source <file>]`
- **Build matching problems**: `quokka build [-d <preset>] --filter <glob> [-j <N>]` (e.g. `'Rad*'`; quote patterns)
- **Build and run (combined)**: `quokka buildrun [-d <preset>] <problem> [<problem> ...] [-j <N>] [--fpe] [--input <file>]`
- **Build and run (filtered)**: `quokka buildrun [-d <preset>] --filter <pattern> [-j <N>]`
- **Run one or more problems**: `quokka run [-d <preset>] <problem> [<problem> ...] [--input <file>] [--fpe]` (`--input` only with one problem)
- **Run all tests**: `quokka run [-d <preset>] [-j <N>]`
- **Run matching tests**: `quokka run [-d <preset>] --filter <regex>` (quote regex/globs to avoid shell expansion)
- **List problems**: `quokka list`
- **Show targets**: `quokka target [-d <preset>]`
- **Clean test output**: `quokka clean`
- **Result summary**: `build`, `run`, and `buildrun` always print final per-target summary lines (`<name> SUCCESS|FAIL|SKIPPED`), so tooling/agents can reliably inspect outcomes by tailing the command output.

Presets: `1d`, `3d`, `1d-debug`, `3d-debug` (sets dimensionality and Release/Debug build type). Default preset is `1d`.

**Without the script (manual):**
- **Configure**: `mkdir -p build/<preset> && cd build/<preset> && cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=<type> -DAMReX_SPACEDIM=<N>`
- **GPU Support**: Add `-DAMReX_GPU_BACKEND=CUDA` (NVIDIA) or `-DAMReX_GPU_BACKEND=HIP` (AMD)
- **Build**: `ninja -j8 <problem>` (from build directory)
- **Run all tests**: `ctest -j8`
- **Run specific test**: `ctest -R TestName`
- **List targets** (returns 1000 targets): `cmake --build . --target help`

- **Test inputs**: Located in `inputs/` directory (`.toml` files)
- **Code formatting**: `clang-format -i file.cpp` (run from `src/` directory)
- **Static analysis**: Use `scripts/tidy.sh build changed` to run clang-tidy on modified files
- **Lint options**: `scripts/tidy.sh build [changed|previous|origin|dev] [--fix]`

## Architecture Overview
- **Main entry**: `src/main.cpp` calls `problem_main()` defined in problem-specific files
- **Core simulation**: `QuokkaSimulation` template class inherits from `AMRSimulation`
- **Physics modules**: Located in `src/hydro/`, `src/radiation/`, `src/cooling/`, `src/chemistry/`
- **Hyperbolic systems**: `HyperbolicSystem` template handles conservation laws and slope limiters
- **Problem definitions**: Each problem in `src/problems/` has `.cpp` files and CMake target
- **I/O and diagnostics**: `src/io/` contains output handling (plotfiles, checkpoints, openPMD)
- **Math utilities**: `src/math/` has interpolation, quadrature, root finding, ODE integration
- **Particles**: `src/particles/` handles stellar particles with accretion, creation, destruction

## Problem Structure
- Each problem directory contains:
  - `test*.cpp`: Implementation with initial conditions and problem-specific physics
  - `CMakeLists.txt`: Defines executable target
- Problems use template specialization pattern for `QuokkaSimulation<ProblemName>`
- Input files (`.toml`) in `inputs/` configure geometry, AMR, physics parameters

## Key Dependencies
- **AMReX**: Underlying AMR framework (external submodule)
- **Microphysics**: Nuclear reaction networks (external submodule)
- **fmt, yaml-cpp**: Formatting and configuration parsing
- **HDF5**: I/O backend
- **OpenPMD-api**: Optional for large-scale output
- **Python**: Optional for analysis tools

## Code Style Guidelines
- Use `.clang-format` from `src/` directory for formatting (LLVM-based style)
- 160 character line limit, indentation with tabs
- Classes use PascalCase (e.g., `QuokkaSimulation`)
- Member variables use camelCase with trailing underscore (e.g., `radiationCflNumber_`)
- Member functions use PascalCase (e.g., `ReadCheckpointFile`)
- Always use curly braces for single statement blocks
- Always use a trailing return type for functions that do not return `void`
- ALWAYS declare variables `const` when they are never modified after initialization.
- Document APIs using Doxygen style comments

## GPU Lambda Safety
- Never capture host pointers inside `AMREX_GPU_DEVICE` lambdas
- Prefer device-safe value types: `amrex::GpuArray` via `geom.ProbLoArray()`, `geom.CellSizeArray()`, `geom.InvCellSizeArray()`
- Never pass `Geometry::ProbLo()/CellSize()` raw pointers into device lambdas; use the array forms
- Never capture raw pointers from `GeometryData` inside GPU lambdas
- Avoid accessing `GeometryData` directly; this is almost never required

## Commit & PR Guidelines
- Use short, imperative commit subjects (e.g., `fix clang-tidy`)
- Group related changes only and rebase onto `development` before opening a PR
- PRs should be focused on a single change and target the `development` branch
