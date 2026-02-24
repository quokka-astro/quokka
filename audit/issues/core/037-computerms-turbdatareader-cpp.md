# computeRms(...): divides by `N` without guarding `N == 0` (`src/turbulence/TurbDataReader.cpp:117`)

## Summary
divides by `N` without guarding `N == 0` (`src/turbulence/TurbDataReader.cpp:117`). Empty turbulence tables yield invalid RMS (`nan`/`inf`).

## Severity
`Medium`

## Affected File
`src/turbulence/TurbDataReader.cpp`

## Affected Function / Symbol
`computeRms(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:536`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
