# AST Audit Coverage

## Objective

Exhaustively review every function from an AST-verified C++ symbol index, excluding vendored dependencies and submodules.

## Evidence Collected

- Configured audit build trees with `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for:
  - `build/audit-1d`
  - `build/audit-3d`
  - `build/audit-2d` as supplemental coverage for 2D-only headers
- Generated `issues/ast-function-index.md` from GCC 15 frontend AST dumps (`-fdump-lang-raw`).
- Processed 165 real Quokka translation-unit commands from the 1D and 3D compile databases with zero failures.
- Added focused AST probes for files not reached by those target compile databases:
  - `src/math/Interpolate2D.hpp`
  - `src/util/CheckNaN.hpp`
  - `src/util/ArrayView_2d.hpp`
  - `src/io/openPMD.cpp`

## Current Gaps

- `src/io/openPMD.hpp` is declaration-only and has no function definitions; `src/io/openPMD.cpp` is covered by a focused AST probe.
- A full `QUOKKA_OPENPMD=ON` configure remains unavailable because the optional OpenPMD/ADIOS2 stack requires ADIOS2 CMake package files that are not installed. The focused probe used the vendored OpenPMD headers, fetched JSON/TOML headers, and a local generated `openPMD/config.hpp` to parse Quokka's wrapper functions without completing the optional OpenPMD build.
- The 2D full-target compile database is not reliable as broad coverage: 54 target syntax checks failed. The successful 2D result used for coverage is limited to the focused `ArrayView_2d.hpp` probe.

## New Finding From AST-Driven Follow-Up

- `issues/radiation-pressure-nan-assertions.md`

## Status

The AST-backed index and follow-up review improved coverage beyond the original regex inventory. Remaining files without AST sections are declaration-only/include-only wrappers or empty aggregation/instantiation files with no function definitions found by follow-up scans.
