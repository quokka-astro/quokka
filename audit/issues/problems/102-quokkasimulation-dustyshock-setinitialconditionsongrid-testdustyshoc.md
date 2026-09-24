# QuokkaSimulation<DustyShock>::setInitialConditionsOnGrid(...): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustyShock/testDustyShock.cpp:70-71`) instead of `grid_elem` geometry

## Summary
uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustyShock/testDustyShock.cpp:70-71`) instead of `grid_elem` geometry. Refined-level IC fills would place the shock using level-0 coordinates.

## Severity
`High`

## Affected File
`src/problems/DustyShock/testDustyShock.cpp`

## Affected Function / Symbol
`QuokkaSimulation<DustyShock>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1395`
- Finding tags: AMR correctness

## Proposed Patch
- Use `grid_elem.dx_` / `grid_elem.prob_lo_` (or level-local geometry) inside IC kernels so refined-level initialization uses the correct coordinates and spacing.
