# Radiation pressure debug checks never reject NaN components

Severity: High

## Explanation

`src/radiation/radiation_system.hpp::RadSystem::ComputeRadPressure` initializes the normal flux and pressure-tensor components to `NAN`, assigns them inside `if constexpr` branches, and then checks them with:

```cpp
AMREX_ASSERT(Fn != NAN);
AMREX_ASSERT(Tnx != NAN);
AMREX_ASSERT(Tny != NAN);
AMREX_ASSERT(Tnz != NAN);
```

NaN never compares equal to itself, so each `x != NAN` expression is always true. If an invalid/non-finite radiation state produces NaNs in the Eddington tensor or flux inputs, the debug checks do not catch it and `result.F` / `result.S` can propagate NaNs into the radiation flux update.

## Patch

```diff
diff --git a/src/radiation/radiation_system.hpp b/src/radiation/radiation_system.hpp
--- a/src/radiation/radiation_system.hpp
+++ b/src/radiation/radiation_system.hpp
@@
-	AMREX_ASSERT(Fn != NAN);
-	AMREX_ASSERT(Tnx != NAN);
-	AMREX_ASSERT(Tny != NAN);
-	AMREX_ASSERT(Tnz != NAN);
+	AMREX_ASSERT(std::isfinite(Fn));
+	AMREX_ASSERT(std::isfinite(Tnx));
+	AMREX_ASSERT(std::isfinite(Tny));
+	AMREX_ASSERT(std::isfinite(Tnz));
 
 	RadPressureResult result{};
```
