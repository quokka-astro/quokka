# AMRSimulation<BeamProblem>::setCustomBoundaryConditions(...): only defines index unpacking for `AMREX_SPACEDIM == 2` or `3` (`src/problems/RadBeam/testRadBeam.cpp:76-82`)

## Summary
only defines index unpacking for `AMREX_SPACEDIM == 2` or `3` (`src/problems/RadBeam/testRadBeam.cpp:76-82`). A 1D build leaves `i/j/k` undefined and this specialization is not dimension-safe.

## Severity
`Low`

## Affected File
`src/problems/RadBeam/testRadBeam.cpp`

## Affected Function / Symbol
`AMRSimulation<BeamProblem>::setCustomBoundaryConditions(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1477`
- Finding tags: portability

## Proposed Patch
- Add compile-time dimension guards or refactor the implementation so indexing and stencil accesses are valid for the supported `AMREX_SPACEDIM` values.
