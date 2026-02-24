# QuokkaSimulation<ShellProblem>::refineGrid(...): refinement indicator unconditionally samples z-neighbors (`state(i,j,k±1,...)`) (`src/problems/RadhydroShell/testRadhydroShell.cpp:277-278`), hard-coding 3D behavior in an unguarded specialization

## Summary
refinement indicator unconditionally samples z-neighbors (`state(i,j,k±1,...)`) (`src/problems/RadhydroShell/testRadhydroShell.cpp:277-278`), hard-coding 3D behavior in an unguarded specialization.

## Severity
`Low`

## Affected File
`src/problems/RadhydroShell/testRadhydroShell.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShellProblem>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1717`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
