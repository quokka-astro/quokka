# AMRSimulation::setDiodeBCLo<...>(...): diode ghost fill copies/reflects only `{rho, mom, E, Eint}` and leaves additional conserved components (e.g

## Summary
diode ghost fill copies/reflects only `{rho, mom, E, Eint}` and leaves additional conserved components (e.g. passive scalars) untouched (`src/simulation.hpp:2642-2653`, `:2689-2694`). This can leave stale ghost values when diode BCs are used with extra hydro state components.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::setDiodeBCLo<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:251`
- Finding tags: correctness

## Proposed Patch
- After populating the core hydro fields, explicitly copy/fill all remaining conserved components (passive scalars and optional extras) in the diode ghost-fill path.

## Why This Is a Bug
The diode ghost fill only writes the core hydro fields explicitly. Any additional cell-centered conserved components (passive scalars, chemistry variables, etc.) are left untouched in ghost zones, so they retain stale values from previous fills/allocations. That creates inconsistent state vectors at the boundary and can contaminate reconstruction/flux calculations.

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
