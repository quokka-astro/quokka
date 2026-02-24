# AdvectionSimulation::computeMaxSignalLocal(int): uses `state_old_cc_[level]` (`src/linear_advection/AdvectionSimulation.hpp:134`) instead of the current state (`state_new_cc_`) when computing CFL speeds, unlike `QuokkaSimulation`

## Summary
uses `state_old_cc_[level]` (`src/linear_advection/AdvectionSimulation.hpp:134`) instead of the current state (`state_new_cc_`) when computing CFL speeds, unlike `QuokkaSimulation`. This can make the timestep estimate use stale data.

## Severity
`Medium`

## Affected File
`src/linear_advection/AdvectionSimulation.hpp`

## Affected Function / Symbol
`AdvectionSimulation::computeMaxSignalLocal(int)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:623`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
