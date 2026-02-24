# QuokkaSimulation<ContactProblem>::computeReferenceSolution(...): in the optional plotting block, vectors are pre-sized to `nx` and then appended with `push_back` inside the loop (`src/problems/HydroContact/testHydroContact.cpp:140-146`, `src/problems/HydroContact/testHydroContact.cpp:150`, `src/problems/HydroContact/testHydroContact.cpp:159-173`)

## Summary
in the optional plotting block, vectors are pre-sized to `nx` and then appended with `push_back` inside the loop (`src/problems/HydroContact/testHydroContact.cpp:140-146`, `src/problems/HydroContact/testHydroContact.cpp:150`, `src/problems/HydroContact/testHydroContact.cpp:159-173`). This doubles vector lengths and prepends default-initialized zeros, corrupting plotted data (test numerics unaffected).

## Severity
`High`

## Affected File
`src/problems/HydroContact/testHydroContact.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ContactProblem>::computeReferenceSolution(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:943`
- Finding tags: plotting-only

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
