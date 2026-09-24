# QuokkaSimulation<QuirkProblem>::computeAfterTimestep(): entropy-jump diagnostic samples at `ilo = ishock_g` (`src/problems/HydroQuirk/testHydroQuirk.cpp:145`), but `ishock_g` is a compile-time constant `0` (`src/problems/HydroQuirk/testHydroQuirk.cpp:69`) and is never updated from the computed shock index in `setInitialConditionsOnGrid(...)` (`src/problems/HydroQuirk/testHydroQuirk.cpp:79-84`)

## Summary
entropy-jump diagnostic samples at `ilo = ishock_g` (`src/problems/HydroQuirk/testHydroQuirk.cpp:145`), but `ishock_g` is a compile-time constant `0` (`src/problems/HydroQuirk/testHydroQuirk.cpp:69`) and is never updated from the computed shock index in `setInitialConditionsOnGrid(...)` (`src/problems/HydroQuirk/testHydroQuirk.cpp:79-84`). The carbuncle diagnostic can monitor the wrong x-location.

## Severity
`Medium`

## Affected File
`src/problems/HydroQuirk/testHydroQuirk.cpp`

## Affected Function / Symbol
`QuokkaSimulation<QuirkProblem>::computeAfterTimestep()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1025`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
