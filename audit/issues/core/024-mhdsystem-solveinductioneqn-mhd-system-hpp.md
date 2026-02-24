# MHDSystem::SolveInductionEqn(...): loops over `w0 = 0..2` unconditionally (`src/hydro/mhd_system.hpp:957`) while operating on `std::array<..., AMREX_SPACEDIM>` containers

## Summary
loops over `w0 = 0..2` unconditionally (`src/hydro/mhd_system.hpp:957`) while operating on `std::array<..., AMREX_SPACEDIM>` containers. This is out-of-bounds for 1D/2D MHD builds and should use `AMREX_SPACEDIM`.

## Severity
`High`

## Affected File
`src/hydro/mhd_system.hpp`

## Affected Function / Symbol
`MHDSystem::SolveInductionEqn(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:398`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.

## Why This Is a Bug
`SolveInductionEqn(...)` iterates `w0 = 0..2` while indexing `std::array<..., AMREX_SPACEDIM>` inputs. In 1D/2D builds this walks off the end of `fc_consVar*` / `ec_emf_mf`. The routine also uses 3D CT indexing conventions, so it should explicitly require 3D (or be fully generalized) instead of silently assuming it.

## Complete Code Patch
```diff
diff --git a/src/hydro/mhd_system.hpp b/src/hydro/mhd_system.hpp
--- a/src/hydro/mhd_system.hpp
+++ b/src/hydro/mhd_system.hpp
@@
 	const BL_PROFILE("MHDSystem::SolveInductionEqn()");
+	static_assert(AMREX_SPACEDIM == 3, "MHDSystem::SolveInductionEqn currently assumes 3D CT indexing.");
 	// compute the total right-hand-side for the MOL integration
@@
-	for (int w0 = 0; w0 < 3; ++w0) {
+	for (int w0 = 0; w0 < AMREX_SPACEDIM; ++w0) {
```
