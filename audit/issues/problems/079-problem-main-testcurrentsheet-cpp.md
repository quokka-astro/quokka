# problem_main(): constructs periodic face-centered BC records `BCs_fc` (`src/problems/CurrentSheet/testCurrentSheet.cpp:111-118`) but then instantiates `QuokkaSimulation<CurrentSheet> sim;` without passing them (`src/problems/CurrentSheet/testCurrentSheet.cpp:120`)

## Summary
constructs periodic face-centered BC records `BCs_fc` (`src/problems/CurrentSheet/testCurrentSheet.cpp:111-118`) but then instantiates `QuokkaSimulation<CurrentSheet> sim;` without passing them (`src/problems/CurrentSheet/testCurrentSheet.cpp:120`). The custom face BC configuration is ignored.

## Severity
`Medium`

## Affected File
`src/problems/CurrentSheet/testCurrentSheet.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1072`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
