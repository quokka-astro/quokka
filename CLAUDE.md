# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++20 codebase.

## Build & Test Commands
- **Build**: `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja && ninja -j6` (keep in mind that `-DAMReX_SPACEDIM` must be set to specify the dimensionality of the code, and that some targets only build for certain dimensionality)
- **GPU Support**: Add `-DAMReX_GPU_BACKEND=CUDA` (NVIDIA) or `-DAMReX_GPU_BACKEND=HIP` (AMD)
- **Run all tests**: `ctest` or `ninja test`
- **Run specific test**: `ctest -R TestName`
- **Exclude tests**: `ctest -E "Pattern*"`
- **List test targets**: `cmake --build . --target help`
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