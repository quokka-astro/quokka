# matplotlibcpp::Plot::update(...): allocates `plot_args` (`src/util/matplotlibcpp.h:1936`) and never decrefs it before returning (`src/util/matplotlibcpp.h:1940-1943`), leaking a Python tuple on each update call

## Summary
allocates `plot_args` (`src/util/matplotlibcpp.h:1936`) and never decrefs it before returning (`src/util/matplotlibcpp.h:1940-1943`), leaking a Python tuple on each update call.

## Severity
`Medium`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::Plot::update(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:819`
- Finding tags: leak

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
