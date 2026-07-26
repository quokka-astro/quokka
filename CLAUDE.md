# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Quokka is a two-moment radiation hydrodynamics code using the piecewise-parabolic method with AMR and subcycling. It's built on AMReX and supports both CPU (MPI+vectorized) and GPU (CUDA/HIP) execution with a single C++20 codebase.

## Build & Test Commands

Use the `quokka` CLI (`scripts/bash/quokka`) to configure, build, and run tests; run `./scripts/bash/bootstrap.sh` once if it is not on your `PATH`. Full command, preset, and lint reference: the **quokka-build** skill (`.claude/skills/quokka-build/SKILL.md`).

## Problem Structure
- Problems use template specialization pattern for `QuokkaSimulation<ProblemName>`

## Code Style Guidelines
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
- **After every commit**, run `./scripts/bash/quokka-pre-commit.sh` to check formatting, YAML validity, merge conflicts, and other CI-enforced hooks
