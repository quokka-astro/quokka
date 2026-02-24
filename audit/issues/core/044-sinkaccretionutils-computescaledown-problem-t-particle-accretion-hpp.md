# SinkAccretionUtils::ComputeScaleDown<problem_t>(...): Jeans-density limiter branch is guarded by `if (accretion_rate_cell > std::numeric_limits<double>::min())` (`src/particles/particle_accretion.hpp:289`), but accretion-zone rates are accumulated as non-positive values (`src/particles/particle_accretion.hpp:237-239`)

## Summary
Jeans-density limiter branch is guarded by `if (accretion_rate_cell > std::numeric_limits<double>::min())` (`src/particles/particle_accretion.hpp:289`), but accretion-zone rates are accumulated as non-positive values (`src/particles/particle_accretion.hpp:237-239`). The intended limiter branch is effectively skipped.

## Severity
`High`

## Affected File
`src/particles/particle_accretion.hpp`

## Affected Function / Symbol
`SinkAccretionUtils::ComputeScaleDown<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:571`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.

## Why This Is a Bug
The accumulated per-cell accretion rates are explicitly non-positive (`rel_accretion_rate <= 0`). Guarding the Jeans limiter with `accretion_rate_cell > min_positive` makes the branch effectively unreachable for accreting cells, so the intended density limiter is skipped even when a cell is in the accretion zone and violates the Jeans cap.

## Complete Code Patch
```diff
diff --git a/src/particles/particle_accretion.hpp b/src/particles/particle_accretion.hpp
--- a/src/particles/particle_accretion.hpp
+++ b/src/particles/particle_accretion.hpp
@@
-			// In the accretion zone, if (1 + accretion_rate_cell) * rho > rho_J, set accretion_rate_cell = rho_J / rho - 1
-			// The condition "accretion_rate_cell > 0.0" is essential as we only want to apply this to the accretion zone. There could be a
+			// In the accretion zone, if (1 + accretion_rate_cell) * rho > rho_J, set accretion_rate_cell = rho_J / rho - 1
+			// The condition "accretion_rate_cell < 0.0" is essential as we only want to apply this to the accretion zone. There could be a
 			// Jeans-violating cell that is not in a accretion zone emerging at the beginning of a step.
-			if (accretion_rate_cell > std::numeric_limits<double>::min()) {
+			if (accretion_rate_cell < -std::numeric_limits<double>::min()) {
```
