# AMRSimulation::RemakeLevel(...): repeats the same face-BC packing bug as `MakeNewLevelFromCoarse` (`BCs_array[idim] = BCs_fc_` at `src/simulation.hpp:2416`), passing flattened BC records where per-direction BC vectors are expected

## Summary
repeats the same face-BC packing bug as `MakeNewLevelFromCoarse` (`BCs_array[idim] = BCs_fc_` at `src/simulation.hpp:2416`), passing flattened BC records where per-direction BC vectors are expected.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::RemakeLevel(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:242`
- Finding tags: correctness

## Proposed Patch
- Slice `BCs_fc_` into per-direction vectors of length `ncomp_per_dim_fc` before calling face-array fill helpers, instead of reusing the full flattened vector for every direction.

## Why This Is a Bug
This repeats the same BC packing error as `MakeNewLevelFromCoarse(...)`: `BCs_fc_` is flattened across face directions, but `FillCoarsePatchFaceArray(...)` expects one per-direction BC vector. Passing the full flattened vector for every direction can apply the wrong BC metadata to face interpolation and physical fills.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 			amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> BCs_array;
 			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
 				int_state_new_fc[idim] = amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
 				int_state_old_fc[idim] = amrex::MultiFab(amrex::convert(ba, amrex::IntVect::TheDimensionVector(idim)), dm, ncomp_per_dim_fc, nghost_fc);
-				BCs_array[idim] = BCs_fc_;
+				BCs_array[idim].clear();
+				for (int n = 0; n < ncomp_per_dim_fc; ++n) {
+					BCs_array[idim].push_back(BCs_fc_[idim * ncomp_per_dim_fc + n]);
+				}
 			}
```
