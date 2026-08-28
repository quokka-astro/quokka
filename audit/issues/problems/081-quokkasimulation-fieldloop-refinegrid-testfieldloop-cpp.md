# QuokkaSimulation<FieldLoop>::refineGrid(...): region-based refinement computes normalized coordinates as `((i+0.5)*dx)/ (phi-plo)` (`src/problems/FieldLoop/testFieldLoop.cpp:148-149`) without subtracting `ProbLo()`

## Summary
region-based refinement computes normalized coordinates as `((i+0.5)*dx)/ (phi-plo)` (`src/problems/FieldLoop/testFieldLoop.cpp:148-149`) without subtracting `ProbLo()`. If the domain lower bound is not zero, the refinement window is shifted/misplaced.

## Severity
`Medium`

## Affected File
`src/problems/FieldLoop/testFieldLoop.cpp`

## Affected Function / Symbol
`QuokkaSimulation<FieldLoop>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1082`
- Finding tags: none

## Proposed Patch
- Normalize coordinates relative to the domain lower bound by subtracting `ProbLo()`/`plo` before scaling to `[0,1]` region-selection coordinates.
