# ParticleChecker::operator()(...): unconditionally indexes `dx[2]` when computing `k_par1/k_par2` (`src/problems/ParticleCreation/testParticleCreation.cpp:137`, `:141`) without a 3D-only guard, so the checker is not dimension-safe for 1D/2D builds

## Summary
unconditionally indexes `dx[2]` when computing `k_par1/k_par2` (`src/problems/ParticleCreation/testParticleCreation.cpp:137`, `:141`) without a 3D-only guard, so the checker is not dimension-safe for 1D/2D builds.

## Severity
`Low`

## Affected File
`src/problems/ParticleCreation/testParticleCreation.cpp`

## Affected Function / Symbol
`ParticleChecker::operator()(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1419`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
