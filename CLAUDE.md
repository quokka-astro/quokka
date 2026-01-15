# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++20 codebase.

## Build & Test Commands
- Run `source ~/rc/qk.rc` to load modules before the first build. 
- **Build test**: `cd /Users/cche/softwares/quokka/quokka/build/clang-3d && ninja -j8 TestName`
- **Run test**: `cd /Users/cche/softwares/quokka/quokka/build/clang-3d && ctest -R TestName`
- **Run test with specific commands**: `cd /Users/cche/softwares/quokka/quokka/tests && ../build/clang-3d/src/problems/TestName/TestName ../inputs/TestName.in tiny_profiler.enabled=0`

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
- Input files (`.in`) in `inputs/` configure geometry, AMR, physics parameters

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
