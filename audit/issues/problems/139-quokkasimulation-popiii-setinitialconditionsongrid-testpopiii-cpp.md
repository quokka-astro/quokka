# QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(...): the IC setup unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/PopIII/testPopIII.cpp:245`, `:261`) without a dimension guard

## Summary
the IC setup unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/PopIII/testPopIII.cpp:245`, `:261`) without a dimension guard.

## Severity
`Low`

## Affected File
`src/problems/PopIII/testPopIII.cpp`

## Affected Function / Symbol
`QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1861`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
