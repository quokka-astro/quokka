# QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(...): IC fill unconditionally uses z-dimension geometry (`src/problems/TallBoxSf/testTallBoxSf.cpp:255`) without compile-time dimension guards

## Summary
IC fill unconditionally uses z-dimension geometry (`src/problems/TallBoxSf/testTallBoxSf.cpp:255`) without compile-time dimension guards.

## Severity
`Low`

## Affected File
`src/problems/TallBoxSf/testTallBoxSf.cpp`

## Affected Function / Symbol
`QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1874`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
