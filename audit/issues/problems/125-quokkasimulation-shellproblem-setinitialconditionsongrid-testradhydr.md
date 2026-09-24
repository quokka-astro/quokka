# QuokkaSimulation<ShellProblem>::setInitialConditionsOnGrid(...): IC kernel and source-center setup unconditionally use z-coordinate geometry (`src/problems/RadhydroShell/testRadhydroShell.cpp:196`, `:212`), so the specialization is not dimension-safe outside 3D

## Summary
IC kernel and source-center setup unconditionally use z-coordinate geometry (`src/problems/RadhydroShell/testRadhydroShell.cpp:196`, `:212`), so the specialization is not dimension-safe outside 3D.

## Severity
`Low`

## Affected File
`src/problems/RadhydroShell/testRadhydroShell.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShellProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1716`
- Finding tags: portability

## Proposed Patch
- Add compile-time dimension guards or refactor the implementation so indexing and stencil accesses are valid for the supported `AMREX_SPACEDIM` values.
