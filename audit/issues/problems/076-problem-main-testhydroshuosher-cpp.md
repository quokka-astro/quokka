# problem_main(): BC initialization loop writes x-direction BCs to `BCs_cc[0]` instead of `BCs_cc[n]` (`src/problems/HydroShuOsher/testHydroShuOsher.cpp:286-289`)

## Summary
BC initialization loop writes x-direction BCs to `BCs_cc[0]` instead of `BCs_cc[n]` (`src/problems/HydroShuOsher/testHydroShuOsher.cpp:286-289`). Only component 0 gets x-boundary settings; other components may retain incorrect/default BCs.

## Severity
`Medium`

## Affected File
`src/problems/HydroShuOsher/testHydroShuOsher.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1009`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
