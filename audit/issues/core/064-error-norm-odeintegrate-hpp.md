# error_norm(...): weighting uses `reltol * y0[i] + abstol[i]` (`src/math/ODEIntegrate.hpp:118`) instead of `reltol * abs(y0[i]) + abstol[i]` (as in standard weighted RMS norms)

## Summary
weighting uses `reltol * y0[i] + abstol[i]` (`src/math/ODEIntegrate.hpp:118`) instead of `reltol * abs(y0[i]) + abstol[i]` (as in standard weighted RMS norms). Negative state values can shrink/cancel the denominator and distort timestep control.

## Severity
`Medium`

## Affected File
`src/math/ODEIntegrate.hpp`

## Affected Function / Symbol
`error_norm(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:832`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
