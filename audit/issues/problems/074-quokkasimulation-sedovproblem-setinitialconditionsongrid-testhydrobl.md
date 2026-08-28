# QuokkaSimulation<SedovProblem>::setInitialConditionsOnGrid(...): after zero-filling all components, it sets `energy_index` but never sets `internalEnergy_index` (`src/problems/HydroBlast3D/testHydroBlast3D.cpp:95-103`)

## Summary
after zero-filling all components, it sets `energy_index` but never sets `internalEnergy_index` (`src/problems/HydroBlast3D/testHydroBlast3D.cpp:95-103`). The dual-energy field remains zero and is inconsistent with the deposited blast energy.

## Severity
`Medium`

## Affected File
`src/problems/HydroBlast3D/testHydroBlast3D.cpp`

## Affected Function / Symbol
`QuokkaSimulation<SedovProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:982`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
