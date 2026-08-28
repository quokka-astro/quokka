# QuokkaSimulation<TheProblem>::addStrangSplitSources(...): source-term kernel unconditionally writes/uses z-components (`posvec[2]`, `GradPhi[2]`) (`src/problems/TallBoxSf/testTallBoxSf.cpp:405`, `:422`) without a dimension guard

## Summary
source-term kernel unconditionally writes/uses z-components (`posvec[2]`, `GradPhi[2]`) (`src/problems/TallBoxSf/testTallBoxSf.cpp:405`, `:422`) without a dimension guard.

## Severity
`Low`

## Affected File
`src/problems/TallBoxSf/testTallBoxSf.cpp`

## Affected Function / Symbol
`QuokkaSimulation<TheProblem>::addStrangSplitSources(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1876`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
