# quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...): kernel computes velocity and energy increments using division by cell density (`src/turbulence/TurbulentDriving.hpp:81`, `:86`, `:90`) without guarding `rho <= 0`

## Summary
kernel computes velocity and energy increments using division by cell density (`src/turbulence/TurbulentDriving.hpp:81`, `:86`, `:90`) without guarding `rho <= 0`.

## Severity
`Medium`

## Affected File
`src/turbulence/TurbulentDriving.hpp`

## Affected Function / Symbol
`quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:542`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
