# matplotlibcpp::Plot: constructor stores `line = PyList_GetItem(res, 0)` (`src/util/matplotlibcpp.h:1915`) without `Py_INCREF`, but later `decref()` unconditionally `Py_DECREF(line)` (`src/util/matplotlibcpp.h:1969-1970`)

## Summary
constructor stores `line = PyList_GetItem(res, 0)` (`src/util/matplotlibcpp.h:1915`) without `Py_INCREF`, but later `decref()` unconditionally `Py_DECREF(line)` (`src/util/matplotlibcpp.h:1969-1970`). `PyList_GetItem` returns a borrowed reference, so ownership semantics are unsafe and can over-decrement.

## Severity
`High`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::Plot`

## Audit Metadata
- Source log: `audit/src-audit-log.md:818`
- Finding tags: refcount ownership

## Proposed Patch
- Fix Python reference ownership: `Py_INCREF` borrowed references that are stored, and `Py_DECREF` only owned references. Audit all adjacent code paths for matching ownership semantics.
