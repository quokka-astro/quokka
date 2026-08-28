# problem_main(): computes custom `BCs_cc` (`src/problems/HydroLeblanc/testHydroLeblanc.cpp:355`) but constructs `QuokkaSimulation<ShocktubeProblem> sim;` without passing them (`src/problems/HydroLeblanc/testHydroLeblanc.cpp:357`)

## Summary
computes custom `BCs_cc` (`src/problems/HydroLeblanc/testHydroLeblanc.cpp:355`) but constructs `QuokkaSimulation<ShocktubeProblem> sim;` without passing them (`src/problems/HydroLeblanc/testHydroLeblanc.cpp:357`). The intended custom BC configuration is ignored.

## Severity
`Medium`

## Affected File
`src/problems/HydroLeblanc/testHydroLeblanc.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1000`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
