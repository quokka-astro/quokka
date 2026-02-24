# HydroSystem::ComputeMagneticEnergy(...): MHD path unconditionally reads `(*cons_fc)[1]` and `(*cons_fc)[2]` (`src/hydro/hydro_system.hpp:668-671`) from a `std::array<..., AMREX_SPACEDIM>`, so it is out-of-bounds for 1D/2D MHD builds

## Summary
MHD path unconditionally reads `(*cons_fc)[1]` and `(*cons_fc)[2]` (`src/hydro/hydro_system.hpp:668-671`) from a `std::array<..., AMREX_SPACEDIM>`, so it is out-of-bounds for 1D/2D MHD builds.

## Severity
`High`

## Affected File
`src/hydro/hydro_system.hpp`

## Affected Function / Symbol
`HydroSystem::ComputeMagneticEnergy(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:385`
- Finding tags: portability

## Proposed Patch
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
