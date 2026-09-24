# AMRSimulation<TheProblem>::setCustomBoundaryConditions(...): diode BC helper unconditionally calls `setDiodeBCLo<2>` / `setDiodeBCHi<2>` (`src/problems/TallBoxSf/testTallBoxSf.cpp:455-456`), so the specialization is not dimension-safe outside 3D

## Summary
diode BC helper unconditionally calls `setDiodeBCLo<2>` / `setDiodeBCHi<2>` (`src/problems/TallBoxSf/testTallBoxSf.cpp:455-456`), so the specialization is not dimension-safe outside 3D.

## Severity
`Low`

## Affected File
`src/problems/TallBoxSf/testTallBoxSf.cpp`

## Affected Function / Symbol
`AMRSimulation<TheProblem>::setCustomBoundaryConditions(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1877`
- Finding tags: portability

## Proposed Patch
- Add compile-time dimension guards or refactor the implementation so indexing and stencil accesses are valid for the supported `AMREX_SPACEDIM` values.
