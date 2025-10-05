# CLAUDE.md

This file provides guidance to AI agents (e.g. claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++17 codebase.

## Build & Test Commands
- **Build**: `cd  /Users/cche/softwares/quokka/quokka/build/clang-3d && ninja -j8 particle_radiation`
- **Run specific test**: `cd /Users/cche/softwares/quokka/quokka/tests && ../build/clang-3d/src/problems/ParticleRadiation/particle_radiation ../inputs/ParticleRadiation.in`

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
  - `CMakeLists.txt`: Defines executable target
- Problems use template specialization pattern for `QuokkaSimulation<ProblemName>`
- Input files (`.in`) in `inputs/` configure geometry, AMR, physics parameters
- Problems should ONLY contain `.cpp` files (no `.hpp` files per recent policy)
