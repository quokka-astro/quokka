# Repository Guidelines

## Project Structure & Module Organization
Core C++20 sources live in `src/`, with physics modules under `hydro/`, `radiation/`, `cooling/`, and `chemistry/`. Shared infrastructure such as `QuokkaSimulation.cpp` sits alongside module code, while scenario drivers compile from `src/problems/`. Runtime inputs (`*.in`) land in `inputs/`, docs in `docs/`, and helper utilities in `scripts/`. Generated builds and intermediates belong in `build/`; regression baselines and plotfiles stay under `tests/` and are tracked through `regression/quokka-tests.ini`.

## Build, Test, and Development Commands
Prefer the `quokka` CLI in `scripts/bash/quokka` for routine workflows. Commands accept `-d <preset>` (`1d`, `3d`, `1d-debug`, `3d-debug`) and default to `1d` when omitted. Configure with `quokka config [-d <preset>]`, adding `--delete` when reconfiguring an existing preset build directory; `config` also accepts extra CMake definitions via repeatable `-D<k>=<v>` flags. Build one or more problems with `quokka build [-d <preset>] <problem> [<problem> ...]` or by glob with `quokka build [-d <preset>] --filter <glob>`. Run problem executables with `quokka run [-d <preset>] <problem>` (optionally `--input <file>` and `--fpe`), run CTest suites with `quokka run [-d <preset>] -j <N>` or `quokka run [-d <preset>] --filter <regex>`, list problems with `quokka list`, inspect targets with `quokka target [-d <preset>]`, and clean generated test outputs with `quokka clean`. Use `--source <file>` on `config/build/buildrun/run/target` when environment setup is required per command.

Raw CMake/Ninja/CTest commands remain supported: configure once per build tree with `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3`, build via `cmake --build build --target all` (or `ninja -C build`), inspect targets using `cmake --build build --target help`, and run tests via `ctest --output-on-failure` or `ctest -R <Pattern>`. Apply clang-tidy to staged changes through `scripts/tidy.sh build changed`.

## Coding Style & Naming Conventions
Follow the repository `.clang-format` (LLVM-derived, 160-column width, tabs at eight spaces) and `.clang-tidy`. Keep headers as `.hpp` and implementations `.cpp`. Prefer PascalCase for classes and methods, camelCase with trailing underscore for data members, and wrap even single statements in braces. Favor trailing return types for non-`void` functions and mark variables `const` whenever possible.

## GPU Lambda Safety
Avoid capturing host pointers inside `AMREX_GPU_DEVICE` lambdas. Prefer device-safe value types like `amrex::GpuArray` (`geom.ProbLoArray()`, `geom.CellSizeArray()`, `geom.InvCellSizeArray()`). Do not pass `Geometry::ProbLo()/CellSize()` raw pointers into device lambdas; use the array forms instead. Never capture raw pointers from `GeometryData` inside GPU lambdas. Avoid accessing `GeometryData` directly; this is almost never required.

## Testing Guidelines
Add a unit test registered with CTest or extend `regression/quokka-tests.ini` with a matching `inputs/` file for new features. Preserve baseline outputs in `tests/` when regression scripts consume them, and document any GPU-only runs by recording the exact commands. Keep numerical changes reproducible with notes on key diagnostics.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects (e.g., `fix clang-tidy`). Group related changes only and rebase onto `development` before opening a PR. Provide PR descriptions covering problem, solution, and validation, cite the `ctest` or benchmark commands run, and link relevant issues. Flag interface or input changes for documentation follow-up and mention required plots or logs when numerics shift.
