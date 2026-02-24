# QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGridFaceVars(...): uses `x1_L = prob_lo[0] + i * dx[0]` for all face directions (`src/problems/BrioWuShockTube/testBrioWuShockTube.cpp:124`) to choose the left/right `B_y` state

## Summary
uses `x1_L = prob_lo[0] + i * dx[0]` for all face directions (`src/problems/BrioWuShockTube/testBrioWuShockTube.cpp:124`) to choose the left/right `B_y` state. For `dir == y` (where x-index is cell-centered, not nodal), this shifts the tangential-field discontinuity by half a cell.

## Severity
`Medium`

## Affected File
`src/problems/BrioWuShockTube/testBrioWuShockTube.cpp`

## Affected Function / Symbol
`QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGridFaceVars(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1060`
- Finding tags: IC alignment

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
