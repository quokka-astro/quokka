# matplotlibcpp::ylim(): same issues as `xlim()` (`src/util/matplotlibcpp.h:1322-1334`): null-check happens after dereference, `args` tuple leaked, raw heap array returned

## Summary
same issues as `xlim()` (`src/util/matplotlibcpp.h:1322-1334`): null-check happens after dereference, `args` tuple leaked, raw heap array returned.

## Severity
`Medium`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::ylim()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:815`
- Finding tags: robustness/leak

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
