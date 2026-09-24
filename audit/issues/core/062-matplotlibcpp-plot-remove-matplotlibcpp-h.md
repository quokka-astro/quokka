# matplotlibcpp::Plot::remove(...): obtains `remove_fct` and allocates `args` (`src/util/matplotlibcpp.h:1955-1957`) but never decrefs either object, leaking Python references

## Summary
obtains `remove_fct` and allocates `args` (`src/util/matplotlibcpp.h:1955-1957`) but never decrefs either object, leaking Python references.

## Severity
`Medium`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::Plot::remove(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:820`
- Finding tags: leak

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
