# QuokkaSimulation<ShadowProblem>::refineGrid(...): refinement indicator unconditionally samples y-neighbors (`state(i, j±1, ...)`) (`src/problems/RadShadow/testRadShadow.cpp:168-169`), making this tagging logic hard-coded for 2D+

## Summary
refinement indicator unconditionally samples y-neighbors (`state(i, j±1, ...)`) (`src/problems/RadShadow/testRadShadow.cpp:168-169`), making this tagging logic hard-coded for 2D+.

## Severity
`Low`

## Affected File
`src/problems/RadShadow/testRadShadow.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShadowProblem>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1623`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
