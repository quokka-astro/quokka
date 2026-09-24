# computeWaveSolution(...): unguarded 3D-only accesses to `prob_lo[2]`/`dx[2]` in CC/FC branches (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:241`, `:249`, `:335-344`) make the helper non-portable to non-3D builds

## Summary
unguarded 3D-only accesses to `prob_lo[2]`/`dx[2]` in CC/FC branches (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:241`, `:249`, `:335-344`) make the helper non-portable to non-3D builds.

## Severity
`Low`

## Affected File
`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp`

## Affected Function / Symbol
`computeWaveSolution(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1282`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
