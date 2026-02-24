# QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(...): uses `geom[0].ProbLoArray()` / `geom[0].CellSizeArray()` (`src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp:70-71`) instead of `grid_elem` geometry

## Summary
uses `geom[0].ProbLoArray()` / `geom[0].CellSizeArray()` (`src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp:70-71`) instead of `grid_elem` geometry. Refined-level IC fills would evaluate Jeans threshold and peak-cell placement with level-0 spacing/coordinates.

## Severity
`High`

## Affected File
`src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp`

## Affected Function / Symbol
`QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1467`
- Finding tags: AMR correctness

## Proposed Patch
- Use `grid_elem.dx_` / `grid_elem.prob_lo_` (or level-local geometry) inside IC kernels so refined-level initialization uses the correct coordinates and spacing.
