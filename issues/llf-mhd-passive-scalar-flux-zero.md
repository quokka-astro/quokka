# MHD LLF fallback drops passive scalar fluxes

Severity: High

## Explanation

`quokka::Riemann::LLF_MHD()` builds `u_L` and `u_R` from the left/right MHD states, then computes passive scalar fluxes from `u_L.scalar[n]` and `u_R.scalar[n]`.

Unlike `HLLD()`, this function never copies `sL.scalar[n]` or `sR.scalar[n]` into `u_L.scalar[n]` and `u_R.scalar[n]`. The `ConsHydro1D` objects are value-initialized, so their scalar arrays remain zero. As a result, every passive scalar component in the MHD LLF flux is zero.

This is high severity because `LLF_MHD` is used by the first-order hydro flux path (`hydroFOFluxFunction`) for MHD. Whenever first-order flux correction or retry logic uses this fallback in an MHD problem with passive scalars, scalar advection can be silently removed at corrected faces.

## Patch

Copy the scalar state into the conserved left/right containers before computing fluxes:

```diff
diff --git a/src/hydro/LLF_mhd.hpp b/src/hydro/LLF_mhd.hpp
--- a/src/hydro/LLF_mhd.hpp
+++ b/src/hydro/LLF_mhd.hpp
@@
 	u_R.Eint = sR.Eint;
 	u_R.by = sR.by;
 	u_R.bz = sR.bz;
+	for (int n = 0; n < N_scalars; ++n) {
+		u_L.scalar[n] = sL.scalar[n];
+		u_R.scalar[n] = sR.scalar[n];
+	}
 
 	//--- Step 2.  Compute wave speeds in L,R states (see Toro eq. 10.43)
```
