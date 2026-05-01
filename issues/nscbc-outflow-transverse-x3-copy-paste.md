# NSCBC outflow X3 copy-paste errors corrupt boundary derivatives and ghosts

Severity: High

## Explanation

`src/hydro/NSCBC_outflow.hpp` has two copy-paste defects in outflow boundary handling:

1. `transverse_xdir_dQ_data()` computes the z-direction transverse derivative, but stores it in `dQ_dy_data` instead of `dQ_dz_data`. Any X1 outflow boundary in 3D therefore feeds the z-derivative contribution through the wrong buffer and leaves `dQ_dz_data` zero.

2. In `setOutflowBoundaryLowOrder()`, the X3 reflecting fallback reads `im3`, `im4`, and `im5` all into `Q_im3`. `Q_im4` and `Q_im5` remain default-zero but are still used to populate deeper ghost cells. For X3 outflow boundaries that switch to reflecting behavior, this can write invalid zero-density/zero-pressure primitive states into ghost zones.

Both affect user-facing hydrodynamic boundary conditions and can silently corrupt boundary states in 3D.

## Patch

```diff
diff --git a/src/hydro/NSCBC_outflow.hpp b/src/hydro/NSCBC_outflow.hpp
--- a/src/hydro/NSCBC_outflow.hpp
+++ b/src/hydro/NSCBC_outflow.hpp
@@
 		if (consVar.contains(ibr, j, k + 1) && consVar.contains(ibr, j, k - 1)) {
 			quokka::valarray<amrex::Real, N> const Qp = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j, k + 1);
 			quokka::valarray<amrex::Real, N> const Qm = HydroSystem<problem_t>::ComputePrimVars(consVar, ibr, j, k - 1);
-			dQ_dy_data = (Qp - Qm) / (2.0 * geom.CellSize(2));
+			dQ_dz_data = (Qp - Qm) / (2.0 * geom.CellSize(2));
 		}
 	}
@@
 		} else if constexpr (DIR == FluxDir::X3) {
 			Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im1);
 			Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im2);
 			Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im3);
-			Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im4);
-			Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im5);
+			Q_im4 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im4);
+			Q_im5 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im5);
 		}
```
