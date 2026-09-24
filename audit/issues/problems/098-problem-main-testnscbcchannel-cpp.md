# problem_main(): component-wise relative error accumulation divides by `U_k` with no zero guard (`src/problems/NscbcChannel/testNscbcChannel.cpp:193-208`)

## Summary
component-wise relative error accumulation divides by `U_k` with no zero guard (`src/problems/NscbcChannel/testNscbcChannel.cpp:193-208`). The passive-scalar reference magnitude is `|s_inflow|`, so `s_inflow = 0` yields division by zero in the error norm.

## Severity
`Medium`

## Affected File
`src/problems/NscbcChannel/testNscbcChannel.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1296`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
