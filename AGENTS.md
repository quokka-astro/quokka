# Repository Guidelines

## Project Structure & Module Organization
Core C++20 sources live in `src/`, with physics modules under `hydro/`, `radiation/`, `cooling/`, and `chemistry/`. Shared infrastructure such as `QuokkaSimulation.cpp` sits alongside module code, while scenario drivers compile from `src/problems/`. Runtime inputs (`*.in`) land in `inputs/`, docs in `docs/`, and helper utilities in `scripts/`. Generated builds and intermediates belong in `build/`; regression baselines and plotfiles stay under `tests/` and are tracked through `regression/quokka-tests.ini`.

## Build, Test, and Development Commands
Configure once per build tree with `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3`. Build everything via `cmake --build build --target all` (or `ninja -C build`). Discover executables using `cmake --build build --target help`. Run the full suite with `ctest --output-on-failure`; narrow runs using `ctest -R <Pattern>` or skip long GPU tests with `ctest -E "MatterEnergyExchange*"`. Apply clang-tidy to staged changes through `scripts/tidy.sh build changed`.

## Coding Style & Naming Conventions
Follow the repository `.clang-format` (LLVM-derived, 160-column width, tabs at eight spaces) and `.clang-tidy`. Keep headers as `.hpp` and implementations `.cpp`. Prefer PascalCase for classes and methods, camelCase with trailing underscore for data members, and wrap even single statements in braces. Favor trailing return types for non-`void` functions and mark variables `const` whenever possible.

## GPU Lambda Safety
Avoid capturing host pointers inside `AMREX_GPU_DEVICE` lambdas. Prefer device-safe value types like `amrex::GpuArray` (`geom.ProbLoArray()`, `geom.CellSizeArray()`, `geom.InvCellSizeArray()`). Do not pass `Geometry::ProbLo()/CellSize()` raw pointers into device lambdas; use the array forms instead. Never capture raw pointers from `GeometryData` inside GPU lambdas. Avoid accessing `GeometryData` directly; this is almost never required.

## Testing Guidelines
Add a unit test registered with CTest or extend `regression/quokka-tests.ini` with a matching `inputs/` file for new features. Preserve baseline outputs in `tests/` when regression scripts consume them, and document any GPU-only runs by recording the exact commands. Keep numerical changes reproducible with notes on key diagnostics.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects (e.g., `fix clang-tidy`). Group related changes only and rebase onto `development` before opening a PR. Provide PR descriptions covering problem, solution, and validation, cite the `ctest` or benchmark commands run, and link relevant issues. Flag interface or input changes for documentation follow-up and mention required plots or logs when numerics shift.
