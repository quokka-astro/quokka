# quokka::ResampledCooling::resampled_cooling_function(...): computes `eint = Eint / rho` and `fastlg(rho/eint)` without guarding `rho > 0` and `eint > 0` (`src/cooling/ResampledCooling.hpp:69-70`)

## Summary
computes `eint = Eint / rho` and `fastlg(rho/eint)` without guarding `rho > 0` and `eint > 0` (`src/cooling/ResampledCooling.hpp:69-70`). Invalid states can produce divide-by-zero/NaNs and invalid table lookups.

## Severity
`Medium`

## Affected File
`src/cooling/ResampledCooling.hpp`

## Affected Function / Symbol
`quokka::ResampledCooling::resampled_cooling_function(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:863`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
