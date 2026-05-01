# Hydro face velocity uses the wrong upwind density

Severity: High

## Explanation

`HydroSystem<problem_t>::ComputeFluxes()` derives a face-centered normal velocity from the mass flux:

```cpp
if (F[density_index] >= 0.) {
    v_norm = F[density_index] / rho_R;
} else {
    v_norm = F[density_index] / rho_L;
}
```

The upwind density is reversed. A positive mass flux comes from the left state and should divide by `rho_L`; a negative mass flux comes from the right state and should divide by `rho_R`.

This face velocity is not just diagnostic. It is accumulated into `avgFaceVel`, used by `HydroSystem::AddInternalEnergyPdV()` for the dual-energy internal-energy update, and passed to tracer particle advection. Contact discontinuities or large density jumps therefore get an incorrect face speed exactly where the density choice matters most.

## Patch

```diff
diff --git a/src/hydro/hydro_system.hpp b/src/hydro/hydro_system.hpp
--- a/src/hydro/hydro_system.hpp
+++ b/src/hydro/hydro_system.hpp
@@
 		// compute face-centered normal velocity
 		double v_norm = 0.0;
 		if (F[density_index] >= 0.) {
-			if (rho_R > 0.) {
-				v_norm = F[density_index] / rho_R;
+			if (rho_L > 0.) {
+				v_norm = F[density_index] / rho_L;
 			}
 		} else {
-			if (rho_L > 0.) {
-				v_norm = F[density_index] / rho_L;
+			if (rho_R > 0.) {
+				v_norm = F[density_index] / rho_R;
 			}
 		}
```
