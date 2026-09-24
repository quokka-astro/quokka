# QuokkaSimulation<DustAdvection3D>::setInitialConditionsOnGrid(...): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustAdvection3D/testDustAdvection3D.cpp:62-63`) instead of `grid_elem` geometry

## Summary
uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustAdvection3D/testDustAdvection3D.cpp:62-63`) instead of `grid_elem` geometry. Refined-level IC fills would use level-0 coordinates and produce incorrect 3D Gaussian placement/width.

## Severity
`High`

## Affected File
`src/problems/DustAdvection3D/testDustAdvection3D.cpp`

## Affected Function / Symbol
`QuokkaSimulation<DustAdvection3D>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1339`
- Finding tags: AMR correctness

## Proposed Patch
- Use `grid_elem.dx_` / `grid_elem.prob_lo_` (or level-local geometry) inside IC kernels so refined-level initialization uses the correct coordinates and spacing.
