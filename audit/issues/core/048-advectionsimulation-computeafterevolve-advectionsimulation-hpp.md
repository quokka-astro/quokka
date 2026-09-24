# AdvectionSimulation::computeAfterEvolve(...): computes `rel_error = err_norm / sol_norm` without guarding `sol_norm == 0` (`src/linear_advection/AdvectionSimulation.hpp:283`)

## Summary
computes `rel_error = err_norm / sol_norm` without guarding `sol_norm == 0` (`src/linear_advection/AdvectionSimulation.hpp:283`). Degenerate zero-reference solutions can produce `inf`/`nan` error norms.

## Severity
`Medium`

## Affected File
`src/linear_advection/AdvectionSimulation.hpp`

## Affected Function / Symbol
`AdvectionSimulation::computeAfterEvolve(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:628`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
