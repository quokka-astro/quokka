# QuokkaSimulation::computeNumberOfRadiationSubsteps(int, Real): computes `dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat)` and `ceil(dt_lev_hydro / dtrad_tmp)` with no guard for `c_hat <= 0` or non-positive `radiationCflNumber_` (`src/QuokkaSimulation.hpp:718-721`), allowing division by zero / invalid substep counts from bad runtime parameters

## Summary
computes `dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat)` and `ceil(dt_lev_hydro / dtrad_tmp)` with no guard for `c_hat <= 0` or non-positive `radiationCflNumber_` (`src/QuokkaSimulation.hpp:718-721`), allowing division by zero / invalid substep counts from bad runtime parameters.

## Severity
`Medium`

## Affected File
`src/QuokkaSimulation.hpp`

## Affected Function / Symbol
`QuokkaSimulation::computeNumberOfRadiationSubsteps(int, Real)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:308`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
