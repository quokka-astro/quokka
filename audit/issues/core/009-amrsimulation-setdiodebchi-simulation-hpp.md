# AMRSimulation::setDiodeBCHi<...>(...): same issue on the upper boundary; only core hydro fields are filled (`src/simulation.hpp:2753-2764`, `:2800-2805`), so passive scalars/other extra conserved components are not boundary-populated

## Summary
same issue on the upper boundary; only core hydro fields are filled (`src/simulation.hpp:2753-2764`, `:2800-2805`), so passive scalars/other extra conserved components are not boundary-populated.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::setDiodeBCHi<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:252`
- Finding tags: correctness

## Proposed Patch
- After populating the core hydro fields, explicitly copy/fill all remaining conserved components (passive scalars and optional extras) in the diode ghost-fill path.

## Why This Is a Bug
The upper-boundary diode fill has the same omission as the lower-boundary version: only the core hydro fields are assigned. Extra conserved components are left uninitialized/stale in ghost cells, which makes the boundary state internally inconsistent and can corrupt any physics module that reads those components.

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 				consVar(i, j, k, HydroSystem<problem_t>::energy_index) =
 				    consVar(i_interior, j_interior, k_interior, HydroSystem<problem_t>::energy_index);
 				consVar(i, j, k, HydroSystem<problem_t>::internalEnergy_index) =
 				    consVar(i_interior, j_interior, k_interior, HydroSystem<problem_t>::internalEnergy_index);
+				for (int n = Physics_NumVars::numHydroVars; n < Physics_Indices<problem_t>::nvarTotal_cc; ++n) {
+					consVar(i, j, k, n) = consVar(i_interior, j_interior, k_interior, n);
+				}
 			} else {
@@
 				consVar(i, j, k, HydroSystem<problem_t>::x3Momentum_index) = (dir == 2) ? mom_normal : x3Mom;
 				consVar(i, j, k, HydroSystem<problem_t>::energy_index) = etot;
 				consVar(i, j, k, HydroSystem<problem_t>::internalEnergy_index) = eint;
+				for (int n = Physics_NumVars::numHydroVars; n < Physics_Indices<problem_t>::nvarTotal_cc; ++n) {
+					consVar(i, j, k, n) = consVar(i_mirror, j_mirror, k_mirror, n);
+				}
 			}
```
