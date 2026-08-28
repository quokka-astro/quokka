# SinkAccretionUtils::UpdateParticleMassAndMomentumInBox<...>(...): accumulates cell velocities via `mom/rho` (`src/particles/particle_accretion.hpp:415-418`) without a non-assert guard for `rho <= 0`

## Summary
accumulates cell velocities via `mom/rho` (`src/particles/particle_accretion.hpp:415-418`) without a non-assert guard for `rho <= 0`.

## Severity
`Medium`

## Affected File
`src/particles/particle_accretion.hpp`

## Affected Function / Symbol
`SinkAccretionUtils::UpdateParticleMassAndMomentumInBox<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:572`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
