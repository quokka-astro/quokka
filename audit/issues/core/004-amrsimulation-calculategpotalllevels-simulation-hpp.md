# AMRSimulation::calculateGpotAllLevels(): the OpenBCSolver branch computes `abstol = abstolPoisson_ * rhs_min` (`src/simulation.hpp:1842`) using the signed minimum RHS value, while the MLMG branch uses `std::abs(rhs_min)` (`src/simulation.hpp:1821`)

## Summary
the OpenBCSolver branch computes `abstol = abstolPoisson_ * rhs_min` (`src/simulation.hpp:1842`) using the signed minimum RHS value, while the MLMG branch uses `std::abs(rhs_min)` (`src/simulation.hpp:1821`). For the common case `rhs_min < 0`, this can pass a negative absolute tolerance to the solver.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::calculateGpotAllLevels()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:229`
- Finding tags: none

## Proposed Patch
- Match the MLMG branch and compute the OpenBCSolver absolute tolerance from `std::abs(rhs_min)` (or a non-negative norm), then assert the final `abstol >= 0`.

## Why This Is a Bug
`abstol` is an absolute tolerance, so it must be non-negative. In the OpenBC solver branch it is computed from the signed minimum RHS value (`rhs_min`), which is commonly negative for Poisson gravity solves. That can pass a negative tolerance into `OpenBCSolver::solve(...)`, producing undefined solver behavior (failed convergence checks, immediate acceptance, or internal assertions depending on AMReX build/runtime checks).

## Complete Code Patch
```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
-				amrex::Real abstol = abstolPoisson_ * rhs_min;
+				amrex::Real abstol = abstolPoisson_ * std::abs(rhs_min);
+				AMREX_ALWAYS_ASSERT(abstol >= 0.0);
 				poissonSolver.solve(amrex::GetVecOfPtrs(phi), amrex::GetVecOfConstPtrs(rhs), reltolPoisson_, abstol);
```
