# computeWaveSolution(...): same unguarded 3D-only accesses (`prob_lo[2]`, `dx[2]`) in CC/FC wave helper (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:225`, `:230`, `:277-282`), which is not dimension-safe for non-3D builds

## Summary
same unguarded 3D-only accesses (`prob_lo[2]`, `dx[2]`) in CC/FC wave helper (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:225`, `:230`, `:277-282`), which is not dimension-safe for non-3D builds.

## Severity
`Low`

## Affected File
`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp`

## Affected Function / Symbol
`computeWaveSolution(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1252`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
