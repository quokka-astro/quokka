# problem_main(): unconditionally configures `BCs_cc[*].setLo/Hi(1, ...)` and extracts a slice with `fextract(..., 1, 0.0)` (`src/problems/RadStreamingY/testRadStreamingY.cpp:160-161`, `:185`), which is invalid for 1D builds without a dimension guard

## Summary
unconditionally configures `BCs_cc[*].setLo/Hi(1, ...)` and extracts a slice with `fextract(..., 1, 0.0)` (`src/problems/RadStreamingY/testRadStreamingY.cpp:160-161`, `:185`), which is invalid for 1D builds without a dimension guard.

## Severity
`Low`

## Affected File
`src/problems/RadStreamingY/testRadStreamingY.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1613`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
