# QuokkaSimulation<DustAdvection>::setInitialConditionsOnGrid(...): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustAdvection/testDustAdvection.cpp:59-60`) instead of `grid_elem.dx_` / `grid_elem.prob_lo_`

## Summary
uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustAdvection/testDustAdvection.cpp:59-60`) instead of `grid_elem.dx_` / `grid_elem.prob_lo_`. If this IC routine runs on refined levels, coordinates are computed with level-0 geometry and the Gaussian profile is misplaced/mis-scaled.

## Severity
`High`

## Affected File
`src/problems/DustAdvection/testDustAdvection.cpp`

## Affected Function / Symbol
`QuokkaSimulation<DustAdvection>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1331`
- Finding tags: AMR correctness

## Proposed Patch
- Use `grid_elem.dx_` / `grid_elem.prob_lo_` (or level-local geometry) inside IC kernels so refined-level initialization uses the correct coordinates and spacing.
