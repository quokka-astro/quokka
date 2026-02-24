# AMRSimulation::calculateGpotAllLevels(): the OpenBCSolver branch computes `abstol = abstolPoisson_ * rhs_min` (`src/simulation.hpp:1842`) using the signed minimum RHS value, while the MLMG branch uses `std::abs(rhs_min)` (`src/simulation.hpp:1821`)

## Summary
the OpenBCSolver branch computes `abstol = abstolPoisson_ * rhs_min` (`src/simulation.hpp:1842`) using the signed minimum RHS value, while the MLMG branch uses `std::abs(rhs_min)` (`src/simulation.hpp:1821`). For the common case `rhs_min < 0`, this can pass a negative absolute tolerance to the solver.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::calculateGpotAllLevels()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:229`
- Finding tags: none

## Proposed Patch
- Match the MLMG branch and compute the OpenBCSolver absolute tolerance from `std::abs(rhs_min)` (or a non-negative norm), then assert the final `abstol >= 0`.
