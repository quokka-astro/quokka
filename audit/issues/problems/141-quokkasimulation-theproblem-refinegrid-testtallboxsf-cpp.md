# QuokkaSimulation<TheProblem>::refineGrid(...): geometrical tagger unconditionally uses z-coordinate geometry (`prob_lo[2]`, `dx[2]`) (`src/problems/TallBoxSf/testTallBoxSf.cpp:145`) without a 3D guard

## Summary
geometrical tagger unconditionally uses z-coordinate geometry (`prob_lo[2]`, `dx[2]`) (`src/problems/TallBoxSf/testTallBoxSf.cpp:145`) without a 3D guard.

## Severity
`Low`

## Affected File
`src/problems/TallBoxSf/testTallBoxSf.cpp`

## Affected Function / Symbol
`QuokkaSimulation<TheProblem>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1871`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
