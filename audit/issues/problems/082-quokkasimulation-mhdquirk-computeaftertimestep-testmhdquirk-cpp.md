# QuokkaSimulation<MHDQuirk>::computeAfterTimestep(): entropy-jump diagnostic samples at `ilo = ishock_g` (`src/problems/MHDQuirk/testMHDQuirk.cpp:160`), but `ishock_g` is a compile-time constant `0` (`src/problems/MHDQuirk/testMHDQuirk.cpp:69`) and is never updated from the computed shock index in `setInitialConditionsOnGrid(...)` (`src/problems/MHDQuirk/testMHDQuirk.cpp:94-100`)

## Summary
entropy-jump diagnostic samples at `ilo = ishock_g` (`src/problems/MHDQuirk/testMHDQuirk.cpp:160`), but `ishock_g` is a compile-time constant `0` (`src/problems/MHDQuirk/testMHDQuirk.cpp:69`) and is never updated from the computed shock index in `setInitialConditionsOnGrid(...)` (`src/problems/MHDQuirk/testMHDQuirk.cpp:94-100`). The carbuncle diagnostic can monitor the wrong x-location.

## Severity
`Medium`

## Affected File
`src/problems/MHDQuirk/testMHDQuirk.cpp`

## Affected Function / Symbol
`QuokkaSimulation<MHDQuirk>::computeAfterTimestep()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1111`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
