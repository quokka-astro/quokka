# QuokkaSimulation<CollapseProblem>::setInitialConditionsOnGrid(...): unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/SphericalCollapse/testSphericalCollapse.cpp:70`, `:75`), so the specialization is not dimension-safe for 1D/2D builds

## Summary
unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/SphericalCollapse/testSphericalCollapse.cpp:70`, `:75`), so the specialization is not dimension-safe for 1D/2D builds.

## Severity
`Low`

## Affected File
`src/problems/SphericalCollapse/testSphericalCollapse.cpp`

## Affected Function / Symbol
`QuokkaSimulation<CollapseProblem>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1816`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
