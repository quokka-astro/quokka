# Sink accretion maps lower-edge particles to the wrong cell

Severity: High

## Explanation

`src/particles/particle_accretion.hpp` computes the cell containing a sink particle with:

```cpp
int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
```

C++ integer conversion truncates toward zero. For positions below the level lower bound, tile lower edge, or valid region by less than one cell, `-0.2` becomes `0` instead of `-1`. Other particle code in this repository uses `amrex::Math::floor` for this same conversion, which is the correct index-space mapping.

The accretion kernels then sample density, momentum, sound speed, and scale-down factors around the wrong stencil center. Near lower boundaries or AMR/ghost transitions this can accrete from the wrong cells and apply the matching mass/momentum update to a different stencil.

## Patch

```diff
diff --git a/src/particles/particle_accretion.hpp b/src/particles/particle_accretion.hpp
--- a/src/particles/particle_accretion.hpp
+++ b/src/particles/particle_accretion.hpp
@@
 #include "AMReX_Array4.H"
 #include "AMReX_BLProfiler.H"
+#include "AMReX_Math.H"
 #include "AMReX_MultiFab.H"
@@
-			int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
-			int iy = static_cast<int>((p.pos(1) - plo[1]) / dx[1]);
-			int iz = static_cast<int>((p.pos(2) - plo[2]) / dx[2]);
+			const int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) / dx[0]));
+			const int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) / dx[1]));
+			const int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) / dx[2]));
@@
-			int ix = static_cast<int>((p.pos(0) - plo[0]) / dx[0]);
-			int iy = static_cast<int>((p.pos(1) - plo[1]) / dx[1]);
-			int iz = static_cast<int>((p.pos(2) - plo[2]) / dx[2]);
+			const int ix = static_cast<int>(amrex::Math::floor((p.pos(0) - plo[0]) / dx[0]));
+			const int iy = static_cast<int>(amrex::Math::floor((p.pos(1) - plo[1]) / dx[1]));
+			const int iz = static_cast<int>(amrex::Math::floor((p.pos(2) - plo[2]) / dx[2]));
```
