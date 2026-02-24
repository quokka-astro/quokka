# problem_main(): Galilean-invariance error norms and pass/fail `status` update are compiled only under `#ifdef HAVE_PYTHON` (`src/problems/SN/testSN.cpp:344-404`), so non-Python builds skip the main invariance validation but still return success except for scalar checks

## Summary
Galilean-invariance error norms and pass/fail `status` update are compiled only under `#ifdef HAVE_PYTHON` (`src/problems/SN/testSN.cpp:344-404`), so non-Python builds skip the main invariance validation but still return success except for scalar checks.

## Severity
`Medium`

## Affected File
`src/problems/SN/testSN.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1833`
- Finding tags: test validity

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
