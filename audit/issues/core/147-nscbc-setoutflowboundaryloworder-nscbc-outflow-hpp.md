# NSCBC::setOutflowBoundaryLowOrder<...>(...): in the `DIR == FluxDir::X3` reflecting fallback branch, `Q_im3` is assigned three times and `Q_im4`/`Q_im5` are never populated (`src/hydro/NSCBC_outflow.hpp:427-432`; repeated assignments at `:430-432`)

## Summary
in the `DIR == FluxDir::X3` reflecting fallback branch, `Q_im3` is assigned three times and `Q_im4`/`Q_im5` are never populated (`src/hydro/NSCBC_outflow.hpp:427-432`; repeated assignments at `:430-432`). This corrupts reflected ghost-state construction for z-boundaries in low-order mode.

## Severity
`High`

## Affected File
`src/hydro/NSCBC_outflow.hpp`

## Affected Function / Symbol
`NSCBC::setOutflowBoundaryLowOrder<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1956`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.

## Why This Is a Bug
In the `DIR == FluxDir::X3` reflecting fallback path, the code overwrites `Q_im3` three times instead of filling `Q_im3`, `Q_im4`, and `Q_im5`. The later reconstruction then uses uninitialized `Q_im4/Q_im5`, corrupting the reflected stencil for z-boundary ghost-state construction.

## Complete Code Patch
```diff
diff --git a/src/hydro/NSCBC_outflow.hpp b/src/hydro/NSCBC_outflow.hpp
--- a/src/hydro/NSCBC_outflow.hpp
+++ b/src/hydro/NSCBC_outflow.hpp
@@
 			} else if constexpr (DIR == FluxDir::X3) {
 				Q_im1 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im1);
 				Q_im2 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im2);
 				Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im3);
-				Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im4);
-				Q_im3 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im5);
+				Q_im4 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im4);
+				Q_im5 = HydroSystem<problem_t>::ComputePrimVars(consVar, i, j, im5);
 			}
```
