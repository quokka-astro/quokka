# ParticleCreator::operator()(...): unconditionally reads/writes z-components (`dx[2]`, `plo[2]`, `p.pos(2)`, `vz`) (`src/problems/ParticleCreation/testParticleCreation.cpp:183`, `:192`) without a 3D-only guard; not dimension-safe for 1D/2D builds

## Summary
unconditionally reads/writes z-components (`dx[2]`, `plo[2]`, `p.pos(2)`, `vz`) (`src/problems/ParticleCreation/testParticleCreation.cpp:183`, `:192`) without a 3D-only guard; not dimension-safe for 1D/2D builds.

## Severity
`Low`

## Affected File
`src/problems/ParticleCreation/testParticleCreation.cpp`

## Affected Function / Symbol
`ParticleCreator::operator()(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1421`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
