# problem_main(): plotting stride uses `int s = nx / 64` (`src/problems/RadForce/testRadForce.cpp:298`) and passes it to `strided_vector_from(...)` (`:308`) without `s >= 1` guard

## Summary
plotting stride uses `int s = nx / 64` (`src/problems/RadForce/testRadForce.cpp:298`) and passes it to `strided_vector_from(...)` (`:308`) without `s >= 1` guard. For `nx < 64` and `HAVE_PYTHON`, this yields `stride == 0`, which is unsafe for `strided_vector_from()`.

## Severity
`Medium`

## Affected File
`src/problems/RadForce/testRadForce.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1513`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
