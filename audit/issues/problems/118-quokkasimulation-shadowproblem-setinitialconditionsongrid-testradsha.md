# QuokkaSimulation<ShadowProblem>::setInitialConditionsOnGrid(...): clump IC construction unconditionally uses `prob_lo[1]` / `dx[1]` (`src/problems/RadShadow/testRadShadow.cpp:127`), so the implementation is not 1D-safe without a dimension guard

## Summary
clump IC construction unconditionally uses `prob_lo[1]` / `dx[1]` (`src/problems/RadShadow/testRadShadow.cpp:127`), so the implementation is not 1D-safe without a dimension guard.

## Severity
`Low`

## Affected File
`src/problems/RadShadow/testRadShadow.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShadowProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1622`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
