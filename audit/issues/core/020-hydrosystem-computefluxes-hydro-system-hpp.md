# HydroSystem::ComputeFluxes<...>(...): face-centered normal velocity uses the opposite-side density when deriving `v_norm` from mass flux (`F[rho] >= 0` divides by `rho_R`, `F[rho] < 0` divides by `rho_L`) at `src/hydro/hydro_system.hpp:1514-1521`

## Summary
face-centered normal velocity uses the opposite-side density when deriving `v_norm` from mass flux (`F[rho] >= 0` divides by `rho_R`, `F[rho] < 0` divides by `rho_L`) at `src/hydro/hydro_system.hpp:1514-1521`. This is inconsistent with the immediately following species upwind logic (`src/hydro/hydro_system.hpp:1525-1544`) and with the linear advection implementation (`src/linear_advection/linear_advection.hpp:190-206`), and can bias tracer advection / dual-energy `div v`.

## Severity
`High`

## Affected File
`src/hydro/hydro_system.hpp`

## Affected Function / Symbol
`HydroSystem::ComputeFluxes<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:386`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
