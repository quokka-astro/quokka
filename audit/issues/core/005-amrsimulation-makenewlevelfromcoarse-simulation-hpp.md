# AMRSimulation::MakeNewLevelFromCoarse(...): constructs per-direction face BC arrays with `BCs_array[idim] = BCs_fc_` (`src/simulation.hpp:2367`) instead of slicing the flattened `BCs_fc_` vector to `ncomp_per_dim_fc` entries

## Summary
constructs per-direction face BC arrays with `BCs_array[idim] = BCs_fc_` (`src/simulation.hpp:2367`) instead of slicing the flattened `BCs_fc_` vector to `ncomp_per_dim_fc` entries. `FillCoarsePatchFaceArray(...)` then receives incorrect BC vector sizes/ordering.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::MakeNewLevelFromCoarse(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:240`
- Finding tags: correctness

## Proposed Patch
- Slice `BCs_fc_` into per-direction vectors of length `ncomp_per_dim_fc` before calling face-array fill helpers, instead of reusing the full flattened vector for every direction.

## Why This Is a Bug
`BCs_fc_` is a flattened face-variable BC vector (all directions concatenated). `FillCoarsePatchFaceArray(...)` expects one BC vector per face direction, each with `ncomp_per_dim_fc` entries. Copying the full flattened vector into every entry (`BCs_array[idim] = BCs_fc_`) gives each direction the wrong size and wrong ordering, so physical boundary handling can read mismatched BC records.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 			amrex::Array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> BCs_array;
 			for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
 				new_mf_array[idim] = &state_new_fc_[level][idim];
 				old_mf_array[idim] = &state_old_fc_[level][idim];
-				BCs_array[idim] = BCs_fc_;
+				BCs_array[idim].clear();
+				for (int n = 0; n < ncomp_per_dim_fc; ++n) {
+					BCs_array[idim].push_back(BCs_fc_[idim * ncomp_per_dim_fc + n]);
+				}
 			}
```
