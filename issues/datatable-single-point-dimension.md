# DataTable accepts one-point dimensions but interpolation requires two points

Severity: High

## Explanation

`src/util/DataTable.hpp` validates table dimension sizes with `sizes_[dim] > 0` and `sizes[dim] > 0`, but interpolation assumes each dimension has at least two points:

```cpp
dcoord_[dim] = (coord_max_[dim] - coord_min_[dim]) / static_cast<amrex::Real>(sizes_[dim] - 1);
...
if (interp.indices[dim] == sizes[dim] - 1) {
	interp.indices[dim] = sizes[dim] - 2;
}
```

For a dimension size of 1, initialization divides by zero. Later interpolation can set the lower index to `-1`, then read before the table. This can happen through programmatic initialization, CSV metadata, or HDF5 metadata because all three paths currently accept `1`.

The table should reject dimensions smaller than two unless a separate constant-table mode is implemented.

## Patch

```diff
diff --git a/src/util/DataTable.hpp b/src/util/DataTable.hpp
--- a/src/util/DataTable.hpp
+++ b/src/util/DataTable.hpp
@@
-			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes_[dim] > 0, std::format("Invalid dimension size {} for dimension {}", sizes_[dim], dim));
+			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes_[dim] > 1,
+							 std::format("Invalid dimension size {} for dimension {}; interpolation requires at least 2 points",
+								     sizes_[dim], dim));
@@
-					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 0,
-									 std::format("Invalid dimension size {} for dimension {}", sizes[dim], dim));
+					AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 1,
+									 std::format("Invalid dimension size {} for dimension {}; interpolation requires at least 2 points",
+										     sizes[dim], dim));
@@
-				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 0, std::format("Invalid dimension size {} for dimension {}", sizes[dim], dim));
+				AMREX_ALWAYS_ASSERT_WITH_MESSAGE(sizes[dim] > 1,
+								 std::format("Invalid dimension size {} for dimension {}; interpolation requires at least 2 points",
+									     sizes[dim], dim));
```
