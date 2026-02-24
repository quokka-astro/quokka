# AMRSimulation::FillCoarsePatch(...): face-centered branch still constructs cell-centered boundary functors (`setBoundaryFunctor`) and never uses `setBoundaryFunctorFaceVar` / `dir` (`src/simulation.hpp:3240-3250`), so custom face-variable physical BCs are skipped on coarse interpolation fills

## Summary
face-centered branch still constructs cell-centered boundary functors (`setBoundaryFunctor`) and never uses `setBoundaryFunctorFaceVar` / `dir` (`src/simulation.hpp:3240-3250`), so custom face-variable physical BCs are skipped on coarse interpolation fills.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::FillCoarsePatch(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:261`
- Finding tags: correctness

## Proposed Patch
- In the face-centered `FillCoarsePatch` branch, construct and pass `setBoundaryFunctorFaceVar` with the correct `dir` so custom face-variable physical BC callbacks run during coarse interpolation fills.
