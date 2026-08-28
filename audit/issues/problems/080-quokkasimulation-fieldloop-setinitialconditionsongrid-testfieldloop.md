# QuokkaSimulation<FieldLoop>::setInitialConditionsOnGrid(...): sets `x3Momentum = rho0 * vz` with `vz = 1.0` (`src/problems/FieldLoop/testFieldLoop.cpp:71`, `:90`) but computes `Ekin` using only `vx^2 + vy^2` (`src/problems/FieldLoop/testFieldLoop.cpp:73`) before forming total energy (`src/problems/FieldLoop/testFieldLoop.cpp:92`)

## Summary
sets `x3Momentum = rho0 * vz` with `vz = 1.0` (`src/problems/FieldLoop/testFieldLoop.cpp:71`, `:90`) but computes `Ekin` using only `vx^2 + vy^2` (`src/problems/FieldLoop/testFieldLoop.cpp:73`) before forming total energy (`src/problems/FieldLoop/testFieldLoop.cpp:92`). The conservative total energy is inconsistent with the initialized momentum state.

## Severity
`Medium`

## Affected File
`src/problems/FieldLoop/testFieldLoop.cpp`

## Affected Function / Symbol
`QuokkaSimulation<FieldLoop>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1080`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
