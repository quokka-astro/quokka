# `interpolate2d` returns a corner value on the upper y boundary

Severity: High

## Explanation

`src/math/Interpolate2D.hpp` handles edge interpolation with special branches when either dimension is clamped to its upper table index. Two of those branches compare `yi`, the lower physical y-coordinate, to `iiy`, an integer table index:

```cpp
} else if (ix == iix && yi != iiy) {
...
} else if (ix != iix && yi == iiy) {
...
} else { // ix == iix && yi == iiy
```

These should compare the y indices (`iy` and `iiy`). As written, queries at the upper y boundary with an interior x usually skip the 1D x-interpolation branch and fall into the final corner case, returning `table(ix, iy)` instead of interpolating between `table(ix, iy)` and `table(ix + 1, iy)`.

This silently corrupts table lookups on a domain boundary.

## Patch

```diff
diff --git a/src/math/Interpolate2D.hpp b/src/math/Interpolate2D.hpp
--- a/src/math/Interpolate2D.hpp
+++ b/src/math/Interpolate2D.hpp
@@
-	} else if (ix == iix && yi != iiy) {
+	} else if (ix == iix && iy != iiy) {
 		const double vol = (y2 - y1);
 		AMREX_ASSERT(vol > 0.);
 		w11 = (y2 - y) / vol;
 		w12 = (y - y1) / vol;
-	} else if (ix != iix && yi == iiy) {
+	} else if (ix != iix && iy == iiy) {
 		const double vol = (x2 - x1);
 		AMREX_ASSERT(vol > 0.);
 		w11 = (x2 - x) / vol;
```
