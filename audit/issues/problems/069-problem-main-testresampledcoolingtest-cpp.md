# problem_main(): reference comparison computes `rel_error = err_norm / sol_norm` without guarding `sol_norm == 0` (`src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp:221-228`)

## Summary
reference comparison computes `rel_error = err_norm / sol_norm` without guarding `sol_norm == 0` (`src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp:221-228`). A degenerate/zero reference dataset would produce `inf`/`nan`.

## Severity
`Medium`

## Affected File
`src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:901`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
