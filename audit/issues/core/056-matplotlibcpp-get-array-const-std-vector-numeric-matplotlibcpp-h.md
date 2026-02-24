# matplotlibcpp::get_array(const std::vector<Numeric>&): in the NumPy path for unsupported element types (`NPY_NOTYPE`), it builds a local temporary `std::vector<double> vd` and returns `PyArray_SimpleNewFromData(..., vd.data())` (`src/util/matplotlibcpp.h:316-320`)

## Summary
in the NumPy path for unsupported element types (`NPY_NOTYPE`), it builds a local temporary `std::vector<double> vd` and returns `PyArray_SimpleNewFromData(..., vd.data())` (`src/util/matplotlibcpp.h:316-320`). The returned NumPy array then points to freed stack storage after the function returns (dangling pointer / use-after-free).

## Severity
`High`

## Affected File
`src/util/matplotlibcpp.h`

## Affected Function / Symbol
`matplotlibcpp::get_array(const std::vector<Numeric>&)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:808`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.

## Why This Is a Bug
In the `NPY_NOTYPE` fallback path, the function builds a local `std::vector<double>` and returns a NumPy array pointing at that temporary buffer. The vector is destroyed on return, so the NumPy object holds a dangling pointer and any later Python access becomes a use-after-free.

## Complete Code Patch
```diff
diff --git a/src/util/matplotlibcpp.h b/src/util/matplotlibcpp.h
--- a/src/util/matplotlibcpp.h
+++ b/src/util/matplotlibcpp.h
@@
 	NPY_TYPES type = select_npy_type<Numeric>::type;
 	if (type == NPY_NOTYPE) {
-		std::vector<double> vd(v.size());
 		npy_intp vsize = v.size();
-		std::copy(v.begin(), v.end(), vd.begin());
-		PyObject *varray = PyArray_SimpleNewFromData(1, &vsize, NPY_DOUBLE, (void *)(vd.data()));
+		PyObject *varray = PyArray_SimpleNew(1, &vsize, NPY_DOUBLE);
+		if (!varray)
+			throw std::runtime_error("NumPy array allocation failed");
+		double *vd = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(varray)));
+		std::copy(v.begin(), v.end(), vd);
 		return varray;
 	}
```
