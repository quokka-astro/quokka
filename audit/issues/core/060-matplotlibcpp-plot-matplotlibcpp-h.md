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

## Why This Is a Bug
`PyList_GetItem` returns a borrowed reference. Storing it in `line` and later calling `Py_DECREF(line)` without first incrementing its refcount means the class decrements an object it never owned. That can underflow the refcount or free the line object while Matplotlib still uses it.

## Complete Code Patch
```diff
diff --git a/src/util/matplotlibcpp.h b/src/util/matplotlibcpp.h
--- a/src/util/matplotlibcpp.h
+++ b/src/util/matplotlibcpp.h
@@
 			if (res) {
 				line = PyList_GetItem(res, 0);
 
-				if (line)
+				if (line) {
+					Py_INCREF(line); // PyList_GetItem returns a borrowed reference
 					set_data_fct = PyObject_GetAttrString(line, "set_data");
-				else
-					Py_DECREF(line);
+				}
 				Py_DECREF(res);
 			}
```
