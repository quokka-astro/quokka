# rk_adaptive_integrate(...): ignores the return code of the initial `rhs(t0, y0, ydot0)` call used for timestep estimation (`src/math/ODEIntegrate.hpp:137`), so an RHS failure can leave `ydot0` invalid and contaminate `dt_guess` / subsequent integration control

## Summary
ignores the return code of the initial `rhs(t0, y0, ydot0)` call used for timestep estimation (`src/math/ODEIntegrate.hpp:137`), so an RHS failure can leave `ydot0` invalid and contaminate `dt_guess` / subsequent integration control.

## Severity
`High`

## Affected File
`src/math/ODEIntegrate.hpp`

## Affected Function / Symbol
`rk_adaptive_integrate(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:833`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
