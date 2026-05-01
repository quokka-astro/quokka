# Hydrostatic atmosphere boundary condition captures GeometryData raw pointers

Severity: High

## Explanation

`src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp` implements a custom boundary condition as an `AMREX_GPU_DEVICE` function. Inside that device function it calls:

```cpp
amrex::Real const *dx = geom.CellSize();
amrex::Real const *prob_lo = geom.ProbLo();
```

Those APIs return raw host-style pointers from `GeometryData`. The repository GPU-safety rules explicitly warn not to use `Geometry::ProbLo()/CellSize()` raw pointers, and not to capture raw pointers from `GeometryData` inside GPU lambdas/device code. On GPU builds this can produce invalid memory access or wrong boundary values.

The safe pattern is to copy geometry data into device-safe value arrays.

## Patch

```diff
diff --git a/src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp b/src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp
--- a/src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp
+++ b/src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp
@@
-	amrex::Real const *dx = geom.CellSize();
-	amrex::Real const *prob_lo = geom.ProbLo();
+	auto const dx = geom.CellSizeArray();
+	auto const prob_lo = geom.ProbLoArray();
 
-	amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
+	amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
```
