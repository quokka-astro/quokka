# PhysicsParticleDescriptor<...>::splitParticles(...): no validation that `splitFactor > 0` (`src/particles/PhysicsParticles.hpp:408-469`)

## Summary
no validation that `splitFactor > 0` (`src/particles/PhysicsParticles.hpp:408-469`). `splitFactor == 0` marks old particles for deletion and creates none; negative values can also overflow `max_new_particles` (`:414`) and corrupt ID/resize logic.

## Severity
`High`

## Affected File
`src/particles/PhysicsParticles.hpp`

## Affected Function / Symbol
`PhysicsParticleDescriptor<...>::splitParticles(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:518`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
