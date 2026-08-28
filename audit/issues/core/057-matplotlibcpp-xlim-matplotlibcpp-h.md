# matplotlibcpp::xlim(): dereferences `res` via `PyTuple_GetItem(res, ...)` before checking whether the Python call failed (`src/util/matplotlibcpp.h:1305-1314`), which can segfault on error

## Summary
dereferences `res` via `PyTuple_GetItem(res, ...)` before checking whether the Python call failed (`src/util/matplotlibcpp.h:1305-1314`), which can segfault on error. It also leaks `args` (never `Py_DECREF(args)`), and returns a raw `new double[2]` requiring caller-managed deletion.

## Severity
`Medium`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::xlim()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:814`
- Finding tags: robustness/leak

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
