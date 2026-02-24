# QuokkaSimulation<TophatProblem>::setInitialConditionsOnGrid(...): tophat geometry classification unconditionally uses y-coordinates (`src/problems/RadTophat/testRadTophat.cpp:214`) and is not 1D-safe without a dimension guard

## Summary
tophat geometry classification unconditionally uses y-coordinates (`src/problems/RadTophat/testRadTophat.cpp:214`) and is not 1D-safe without a dimension guard.

## Severity
`Low`

## Affected File
`src/problems/RadTophat/testRadTophat.cpp`

## Affected Function / Symbol
`QuokkaSimulation<TophatProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1635`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
