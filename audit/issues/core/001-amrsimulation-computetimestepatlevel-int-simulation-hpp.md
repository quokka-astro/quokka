# AMRSimulation::computeTimestepAtLevel(int): `hydro_dt` is computed as `dx_min / domain_signal_max` without guarding `domain_signal_max <= 0` or non-finite values (`src/simulation.hpp:1171`)

## Summary
`hydro_dt` is computed as `dx_min / domain_signal_max` without guarding `domain_signal_max <= 0` or non-finite values (`src/simulation.hpp:1171`). A zero/NaN signal speed can produce `inf`/`nan` timestep and contaminate later timestep logic.

## Severity
`Medium`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::computeTimestepAtLevel(int)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:223`
- Finding tags: none

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
