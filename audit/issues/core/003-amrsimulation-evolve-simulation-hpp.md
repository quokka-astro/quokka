# AMRSimulation::evolve(): time sync assert divides by `cur_time` (`src/simulation.hpp:1459`)

## Summary
time sync assert divides by `cur_time` (`src/simulation.hpp:1459`). If a zero timestep is produced (e.g., pathological CFL calculation), this assertion becomes `0/0` or `x/0` and fails non-diagnostically.

## Severity
`Medium`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::evolve()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:228`
- Finding tags: none

## Proposed Patch
- Rework the time-sync assertion to avoid dividing by `cur_time`; use an absolute tolerance when `cur_time == 0` and a relative check otherwise.
