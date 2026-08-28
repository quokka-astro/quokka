# HydroSystem::ComputeMagneticEnergy(...): MHD path unconditionally reads `(*cons_fc)[1]` and `(*cons_fc)[2]` (`src/hydro/hydro_system.hpp:668-671`) from a `std::array<..., AMREX_SPACEDIM>`, so it is out-of-bounds for 1D/2D MHD builds

## Summary
MHD path unconditionally reads `(*cons_fc)[1]` and `(*cons_fc)[2]` (`src/hydro/hydro_system.hpp:668-671`) from a `std::array<..., AMREX_SPACEDIM>`, so it is out-of-bounds for 1D/2D MHD builds.

## Severity
`Low`

## Affected File
`src/hydro/hydro_system.hpp`

## Affected Function / Symbol
`HydroSystem::ComputeMagneticEnergy(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:385`
- Finding tags: portability

## Proposed Patch
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.

## Why This Is a Bug
`ComputeMagneticEnergy(...)` unconditionally reads face arrays `(*cons_fc)[1]` and `(*cons_fc)[2]`, but the container size is `AMREX_SPACEDIM`. In 1D/2D MHD builds this reads past the array bounds and returns undefined values (or crashes) while computing magnetic energy.

## Complete Code Patch
```diff
diff --git a/src/hydro/hydro_system.hpp b/src/hydro/hydro_system.hpp
--- a/src/hydro/hydro_system.hpp
+++ b/src/hydro/hydro_system.hpp
@@
 			const auto bx1_m = (*cons_fc)[0](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
 			const auto bx1_p = (*cons_fc)[0](i + 1, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
-			const auto bx2_m = (*cons_fc)[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
-			const auto bx2_p = (*cons_fc)[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
-			const auto bx3_m = (*cons_fc)[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
-			const auto bx3_p = (*cons_fc)[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
 			const auto bx1 = 0.5 * (bx1_m + bx1_p);
-			const auto bx2 = 0.5 * (bx2_m + bx2_p);
-			const auto bx3 = 0.5 * (bx3_m + bx3_p);
+			amrex::Real bx2 = 0.0;
+			amrex::Real bx3 = 0.0;
+			if constexpr (AMREX_SPACEDIM >= 2) {
+				const auto bx2_m = (*cons_fc)[1](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
+				const auto bx2_p = (*cons_fc)[1](i, j + 1, k, Physics_Indices<problem_t>::mhdFirstIndex);
+				bx2 = 0.5 * (bx2_m + bx2_p);
+			}
+			if constexpr (AMREX_SPACEDIM == 3) {
+				const auto bx3_m = (*cons_fc)[2](i, j, k, Physics_Indices<problem_t>::mhdFirstIndex);
+				const auto bx3_p = (*cons_fc)[2](i, j, k + 1, Physics_Indices<problem_t>::mhdFirstIndex);
+				bx3 = 0.5 * (bx3_m + bx3_p);
+			}
 			const auto b_sq = bx1 * bx1 + bx2 * bx2 + bx3 * bx3;
```
