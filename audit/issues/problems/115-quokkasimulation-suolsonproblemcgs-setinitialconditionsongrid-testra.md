# QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(...): `state_cc(..., radEnergy_index)` is assigned twice consecutively (`src/problems/RadMarshakCGS/testRadMarshakCGS.cpp:183-184`)

## Summary
`state_cc(..., radEnergy_index)` is assigned twice consecutively (`src/problems/RadMarshakCGS/testRadMarshakCGS.cpp:183-184`). The duplicate write is likely harmless but indicates a copy/paste bug in the IC kernel.

## Severity
`Medium`

## Affected File
`src/problems/RadMarshakCGS/testRadMarshakCGS.cpp`

## Affected Function / Symbol
`QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1555`
- Finding tags: code quality/correctness hygiene

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
