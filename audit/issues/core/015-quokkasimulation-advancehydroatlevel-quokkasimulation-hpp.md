# QuokkaSimulation::advanceHydroAtLevel(...): Stage-2 face-centered ghost fill uses `time` instead of `time + dt_lev` (`src/QuokkaSimulation.hpp:2309`) while the cell-centered fill uses `time + dt_lev` (`src/QuokkaSimulation.hpp:2305`)

## Summary
Stage-2 face-centered ghost fill uses `time` instead of `time + dt_lev` (`src/QuokkaSimulation.hpp:2309`) while the cell-centered fill uses `time + dt_lev` (`src/QuokkaSimulation.hpp:2305`). This can apply inconsistent boundary states for time-dependent MHD boundary conditions during the RK2 corrector stage.

## Severity
`Medium`

## Affected File
`src/QuokkaSimulation.hpp`

## Affected Function / Symbol
`QuokkaSimulation::advanceHydroAtLevel(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:331`
- Finding tags: none

## Proposed Patch
- Use `time + dt_lev` consistently for both cell-centered and face-centered RK2 stage-2 boundary fills to keep time-dependent BCs synchronized.
