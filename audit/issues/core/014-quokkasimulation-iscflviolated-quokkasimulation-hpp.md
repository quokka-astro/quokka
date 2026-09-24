# QuokkaSimulation::isCflViolated(...): computes `dt_cfl = cflNumber_ * (dx_min / max_signal)` without guarding zero/non-finite `max_signal` (`src/QuokkaSimulation.hpp:2037`), allowing `inf`/`nan` CFL thresholds and unreliable retry acceptance

## Summary
computes `dt_cfl = cflNumber_ * (dx_min / max_signal)` without guarding zero/non-finite `max_signal` (`src/QuokkaSimulation.hpp:2037`), allowing `inf`/`nan` CFL thresholds and unreliable retry acceptance.

## Severity
`Medium`

## Affected File
`src/QuokkaSimulation.hpp`

## Affected Function / Symbol
`QuokkaSimulation::isCflViolated(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:330`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
