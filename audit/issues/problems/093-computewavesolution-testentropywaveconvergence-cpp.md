# computeWaveSolution(...): unguarded accesses to `prob_lo[2]` and `dx[2]` in CC/FC branches (`src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp:168`, `:173`, `:213-222`) make the helper non-portable to non-3D builds

## Summary
unguarded accesses to `prob_lo[2]` and `dx[2]` in CC/FC branches (`src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp:168`, `:173`, `:213-222`) make the helper non-portable to non-3D builds.

## Severity
`Low`

## Affected File
`src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp`

## Affected Function / Symbol
`computeWaveSolution(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1262`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
