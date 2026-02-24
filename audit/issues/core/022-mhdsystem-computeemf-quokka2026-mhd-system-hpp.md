# MHDSystem::ComputeEMF_Quokka2026(...): hard-codes 3D indexing (`fcx_mf_vel[2]`, `fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:371-385` and `:475-477`, causing out-of-bounds access in 1D/2D MHD builds

## Summary
hard-codes 3D indexing (`fcx_mf_vel[2]`, `fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:371-385` and `:475-477`, causing out-of-bounds access in 1D/2D MHD builds.

## Severity
`High`

## Affected File
`src/hydro/mhd_system.hpp`

## Affected Function / Symbol
`MHDSystem::ComputeEMF_Quokka2026(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:392`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.

## Why This Is a Bug
`ComputeEMF_Quokka2026(...)` mixes `std::array<..., AMREX_SPACEDIM>` inputs with hard-coded 3D indexing (`[2]`, arrays of size `3`, `iedge < 3`). In lower-dimensional MHD builds this compiles into out-of-bounds accesses. The implementation is fundamentally 3D CT logic, so the safe fix is to make that assumption explicit and remove hidden OOB hazards.

## Complete Code Patch
```diff
diff --git a/src/hydro/mhd_system.hpp b/src/hydro/mhd_system.hpp
--- a/src/hydro/mhd_system.hpp
+++ b/src/hydro/mhd_system.hpp
@@
 	const BL_PROFILE("MHDSystem::ComputeEMF_Quokka2026()");
+	static_assert(AMREX_SPACEDIM == 3, "MHDSystem::ComputeEMF_Quokka2026 currently assumes 3D CT indexing.");
@@
-			std::array<amrex::FArrayBox, 3> fc_fabs_Ux = {
+			std::array<amrex::FArrayBox, AMREX_SPACEDIM> fc_fabs_Ux = {
@@
-			std::array<amrex::FArrayBox, 3> fc_fabs_Bx = {
+			std::array<amrex::FArrayBox, AMREX_SPACEDIM> fc_fabs_Bx = {
@@
-			for (int iedge = 0; iedge < 3; ++iedge) {
+			for (int iedge = 0; iedge < AMREX_SPACEDIM; ++iedge) {
@@
-				std::array<amrex::Array4<const amrex::Real>, 3> const fspds = {fcx_mf_fspds[0].const_array(mfi), fcx_mf_fspds[1].const_array(mfi),
-								       fcx_mf_fspds[2].const_array(mfi)};
+				std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> const fspds = {fcx_mf_fspds[0].const_array(mfi),
+										       fcx_mf_fspds[1].const_array(mfi),
+										       fcx_mf_fspds[2].const_array(mfi)};
 				MHDSystem<problem_t>::AverageEMF(E2_ave, ec_fabs_E_Q, box_ec, field_w_indices, fspds, ec_fabs_Bi_ieside, emf_avg_scheme);
```
