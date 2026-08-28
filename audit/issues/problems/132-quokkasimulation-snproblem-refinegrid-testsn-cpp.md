# QuokkaSimulation<SNProblem>::refineGrid(...): normalized coordinates omit subtraction of `ProbLo()` (`src/problems/SN/testSN.cpp:198-200`), so the selected subregion shifts if domain lower bounds are nonzero

## Summary
normalized coordinates omit subtraction of `ProbLo()` (`src/problems/SN/testSN.cpp:198-200`), so the selected subregion shifts if domain lower bounds are nonzero.

## Severity
`Medium`

## Affected File
`src/problems/SN/testSN.cpp`

## Affected Function / Symbol
`QuokkaSimulation<SNProblem>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1830`
- Finding tags: AMR region selection

## Proposed Patch
- Normalize coordinates relative to the domain lower bound by subtracting `ProbLo()`/`plo` before scaling to `[0,1]` region-selection coordinates.
