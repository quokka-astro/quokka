# QuokkaSimulation<BlastProblem>::setInitialConditionsOnGrid(...): initializes `energy_index` but never initializes `internalEnergy_index` (`src/problems/HydroBlast2D/testHydroBlast2D.cpp:92-103`), and this routine does not zero-fill all components first

## Summary
initializes `energy_index` but never initializes `internalEnergy_index` (`src/problems/HydroBlast2D/testHydroBlast2D.cpp:92-103`), and this routine does not zero-fill all components first. The dual-energy variable can remain garbage/undefined at startup.

## Severity
`Medium`

## Affected File
`src/problems/HydroBlast2D/testHydroBlast2D.cpp`

## Affected Function / Symbol
`QuokkaSimulation<BlastProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:973`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
