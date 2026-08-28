# quokka::PeHeatingFromSfh(...): divides by `sf_area_kpc2` without validating it is positive/non-zero (`src/cooling/PhotoelectricHeating.hpp:77`)

## Summary
divides by `sf_area_kpc2` without validating it is positive/non-zero (`src/cooling/PhotoelectricHeating.hpp:77`). A zero area configuration will produce `inf`/`nan` heating rates.

## Severity
`Medium`

## Affected File
`src/cooling/PhotoelectricHeating.hpp`

## Affected Function / Symbol
`quokka::PeHeatingFromSfh(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:857`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
