# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++20 codebase.

## Build & Test Commands

The `scripts/bash/quokka` script is the recommended way to configure, build, and run tests. Make sure it exists in your `PATH`; if not, install it once by copying it to a directory on your `PATH` (e.g. `~/.local/bin/`). All commands accept `--root <path>` to specify the repo root when not running from it.

- **Configure**: `quokka config <preset>` — runs CMake with the correct dimensionality and build type
- **Build a problem**: `quokka build <preset> <problem> [-j <N>]`
- **Run a problem**: `quokka run <preset> <problem> [--input <file>] [--fpe]`
- **Run all tests**: `quokka run <preset> [-j <N>]`
- **Run matching tests**: `quokka run <preset> --filter <regex>`
- **List problems**: `quokka list <preset>`
- **Show targets**: `quokka target <preset>`
- **Clean test output**: `quokka clean`

Presets: `1d`, `3d`, `1d-debug`, `3d-debug` (sets dimensionality and Release/Debug build type).

**Without the script (manual):**
- **Configure**: `mkdir -p build/<preset> && cd build/<preset> && cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=<type> -DAMReX_SPACEDIM=<N>`
- **GPU Support**: Add `-DAMReX_GPU_BACKEND=CUDA` (NVIDIA) or `-DAMReX_GPU_BACKEND=HIP` (AMD)
- **Build**: `ninja -j8 <problem>` (from build directory)
- **Run all tests**: `ctest -j8`
- **Run specific test**: `ctest -R TestName`
- **List targets** (returns 1000 targets): `cmake --build . --target help` 

- **Test inputs**: Located in `inputs/` directory (`.in` files)
- **Code formatting**: `clang-format -i file.cpp` (run from `src/` directory)
- **Static analysis**: Use `scripts/tidy.sh build changed` to run clang-tidy on modified files
- **Lint options**: `scripts/tidy.sh build [changed|previous|origin|dev] [--fix]`

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
- PRs should be focused on a single change and target the `development` branch