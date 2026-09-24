# SinkAccretionUtils::compute_Mdot_and_r_K<problem_t>(...): computes `rho_infty = sum_rho / n_cells`, `vx_grid = sum_px / sum_rho`, and `cs_infty = sum_cs / sum_rho` (`src/particles/particle_accretion.hpp:95-103`) before guarding zero counts/mass

## Summary
computes `rho_infty = sum_rho / n_cells`, `vx_grid = sum_px / sum_rho`, and `cs_infty = sum_cs / sum_rho` (`src/particles/particle_accretion.hpp:95-103`) before guarding zero counts/mass. Pathological empty/zero-density stencils can produce `inf`/`nan`.

## Severity
`Medium`

## Affected File
`src/particles/particle_accretion.hpp`

## Affected Function / Symbol
`SinkAccretionUtils::compute_Mdot_and_r_K<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:568`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
