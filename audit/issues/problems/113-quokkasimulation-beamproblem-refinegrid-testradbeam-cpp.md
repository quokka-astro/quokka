# QuokkaSimulation<BeamProblem>::refineGrid(...): refinement indicator unconditionally accesses y-neighbors (`state(i, j±1, ...)`) (`src/problems/RadBeam/testRadBeam.cpp:257-258`), so the implementation is hard-coded for 2D+ and not 1D-safe

## Summary
refinement indicator unconditionally accesses y-neighbors (`state(i, j±1, ...)`) (`src/problems/RadBeam/testRadBeam.cpp:257-258`), so the implementation is hard-coded for 2D+ and not 1D-safe.

## Severity
`Low`

## Affected File
`src/problems/RadBeam/testRadBeam.cpp`

## Affected Function / Symbol
`QuokkaSimulation<BeamProblem>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1480`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
