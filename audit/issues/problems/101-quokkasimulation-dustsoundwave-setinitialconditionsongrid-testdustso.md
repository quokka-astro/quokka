# QuokkaSimulation<DustSoundwave>::setInitialConditionsOnGrid(...): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustSoundwave/testDustSoundwave.cpp:121-122`) rather than `grid_elem.dx_` / `grid_elem.prob_lo_`

## Summary
uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustSoundwave/testDustSoundwave.cpp:121-122`) rather than `grid_elem.dx_` / `grid_elem.prob_lo_`. Refined-level IC fills would use level-0 geometry and mis-phase the wave.

## Severity
`High`

## Affected File
`src/problems/DustSoundwave/testDustSoundwave.cpp`

## Affected Function / Symbol
`QuokkaSimulation<DustSoundwave>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1387`
- Finding tags: AMR correctness

## Proposed Patch
- Use `grid_elem.dx_` / `grid_elem.prob_lo_` (or level-local geometry) inside IC kernels so refined-level initialization uses the correct coordinates and spacing.
