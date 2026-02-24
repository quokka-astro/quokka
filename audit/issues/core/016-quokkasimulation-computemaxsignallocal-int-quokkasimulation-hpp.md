# QuokkaSimulation::computeMaxSignalLocal(int): MHD face arrays are filled with `for (int idim = 0; idim < 3; ++idim)` (`src/QuokkaSimulation.hpp:735`) even though the container type is `std::array<..., AMREX_SPACEDIM>`

## Summary
MHD face arrays are filled with `for (int idim = 0; idim < 3; ++idim)` (`src/QuokkaSimulation.hpp:735`) even though the container type is `std::array<..., AMREX_SPACEDIM>`. This is out-of-bounds for 1D/2D MHD builds.

## Severity
`High`

## Affected File
`src/QuokkaSimulation.hpp`

## Affected Function / Symbol
`QuokkaSimulation::computeMaxSignalLocal(int)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:332`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.

## Why This Is a Bug
`stateNew_fc` is a `std::array<..., AMREX_SPACEDIM>`, but the code writes indices `0,1,2` unconditionally. In 1D or 2D MHD builds that writes past the end of the array and causes undefined behavior before `HydroSystem::ComputeMaxSignalSpeed(...)` is even called.

## Complete Code Patch
```diff
diff --git a/src/QuokkaSimulation.hpp b/src/QuokkaSimulation.hpp
--- a/src/QuokkaSimulation.hpp
+++ b/src/QuokkaSimulation.hpp
@@
 			std::array<amrex::Array4<const amrex::Real>, AMREX_SPACEDIM> stateNew_fc;
 			if constexpr (Physics_Traits<problem_t>::is_mhd_enabled) {
-				for (int idim = 0; idim < 3; ++idim) {
+				for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
 					stateNew_fc[idim] = state_new_fc_[level][idim].const_array(iter);
 				}
 			}
```
