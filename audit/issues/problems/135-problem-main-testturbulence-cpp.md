# problem_main(): the dispersion check reads `sim.turbParams_["target_vdisp"]` via `std::stod(...)` and divides by `target_vdisp` without validating presence/nonzero value (`src/problems/Turbulence/testTurbulence.cpp:133-134`), so malformed or zero input can throw or produce invalid relative error

## Summary
the dispersion check reads `sim.turbParams_["target_vdisp"]` via `std::stod(...)` and divides by `target_vdisp` without validating presence/nonzero value (`src/problems/Turbulence/testTurbulence.cpp:133-134`), so malformed or zero input can throw or produce invalid relative error.

## Severity
`Medium`

## Affected File
`src/problems/Turbulence/testTurbulence.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1843`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
