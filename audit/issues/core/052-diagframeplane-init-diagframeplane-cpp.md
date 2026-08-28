# DiagFramePlane::init(...): filter-warning condition is reversed

## Summary
filter-warning condition is reversed. It prints "filters ... will be discarded" only when `m_filters.empty()` (`src/io/DiagFramePlane.cpp:47`), and does not clear filters when they are actually present. Users specifying filters get no warning despite filters being unsupported here.

## Severity
`Low`

## Affected File
`src/io/DiagFramePlane.cpp`

## Affected Function / Symbol
`DiagFramePlane::init(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:702`
- Finding tags: low

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
