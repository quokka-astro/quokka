# QuokkaSimulation::replaceEMFs(...): loops `for (int iedge = 0; iedge < 3; ++iedge)` over `std::array<amrex::MultiFab, AMREX_SPACEDIM>` (`src/QuokkaSimulation.hpp:2507`), which is out-of-bounds for 1D/2D MHD builds

## Summary
loops `for (int iedge = 0; iedge < 3; ++iedge)` over `std::array<amrex::MultiFab, AMREX_SPACEDIM>` (`src/QuokkaSimulation.hpp:2507`), which is out-of-bounds for 1D/2D MHD builds.

## Severity
`Low`

## Affected File
`src/QuokkaSimulation.hpp`

## Affected Function / Symbol
`QuokkaSimulation::replaceEMFs(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:333`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.

## Why This Is a Bug
`emf_components` and `FO_emf_components` are `std::array<..., AMREX_SPACEDIM>`. Looping to `3` unconditionally dereferences nonexistent entries in 1D/2D builds, causing out-of-bounds accesses when replacing EMFs in flagged cells.

## Complete Code Patch
```diff
diff --git a/src/QuokkaSimulation.hpp b/src/QuokkaSimulation.hpp
--- a/src/QuokkaSimulation.hpp
+++ b/src/QuokkaSimulation.hpp
@@
-	for (int iedge = 0; iedge < 3; ++iedge) { // loop over edges
+	for (int iedge = 0; iedge < AMREX_SPACEDIM; ++iedge) { // loop over edges
```
