# SNFeedbackUtils::addCompositeBufferToState(...): computes `d_e_int_d_rho = e_int / rho` (`src/particles/particle_deposition.hpp:511`) without guarding `rho <= 0`

## Summary
computes `d_e_int_d_rho = e_int / rho` (`src/particles/particle_deposition.hpp:511`) without guarding `rho <= 0`. If a feedback-affected cell has zero/non-positive gas density, this generates invalid energy updates.

## Severity
`Medium`

## Affected File
`src/particles/particle_deposition.hpp`

## Affected Function / Symbol
`SNFeedbackUtils::addCompositeBufferToState(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:505`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
