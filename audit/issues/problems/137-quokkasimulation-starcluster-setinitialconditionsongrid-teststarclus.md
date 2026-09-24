# QuokkaSimulation<StarCluster>::setInitialConditionsOnGrid(...): the IC kernel unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/StarCluster/testStarCluster.cpp:136`, `:152`) and `dvz`, so the specialization is not dimension-safe for 1D/2D builds

## Summary
the IC kernel unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/StarCluster/testStarCluster.cpp:136`, `:152`) and `dvz`, so the specialization is not dimension-safe for 1D/2D builds.

## Severity
`Low`

## Affected File
`src/problems/StarCluster/testStarCluster.cpp`

## Affected Function / Symbol
`QuokkaSimulation<StarCluster>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1851`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
