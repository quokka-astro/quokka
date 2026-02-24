# strided_vector_from(...): no validation for `stride <= 0` (`src/util/ArrayUtil.hpp:16`)

## Summary
no validation for `stride <= 0` (`src/util/ArrayUtil.hpp:16`). A zero stride causes an infinite loop; a negative stride underflows the unsigned loop index progression.

## Severity
`Medium`

## Affected File
`src/util/ArrayUtil.hpp`

## Affected Function / Symbol
`strided_vector_from(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:750`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
