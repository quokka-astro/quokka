# matplotlibcpp::subplot2grid(...): `PyTuple_SetItem(args, 0/1, shape/loc)` steals references, but the code then manually decrefs `shape` and `loc` (`src/util/matplotlibcpp.h:1480-1491`), causing refcount underflow / premature free / double-decref risk

## Summary
`PyTuple_SetItem(args, 0/1, shape/loc)` steals references, but the code then manually decrefs `shape` and `loc` (`src/util/matplotlibcpp.h:1480-1491`), causing refcount underflow / premature free / double-decref risk.

## Severity
`High`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::subplot2grid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:816`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
