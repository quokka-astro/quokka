# HydroSystem::ComputeMaxSignalSpeed(...): MHD branch unconditionally reads `cons_fc[1]` and `cons_fc[2]` (`src/hydro/hydro_system.hpp:409-412`) from a `std::array<..., AMREX_SPACEDIM>`, causing out-of-bounds access for 1D/2D MHD builds

## Summary
MHD branch unconditionally reads `cons_fc[1]` and `cons_fc[2]` (`src/hydro/hydro_system.hpp:409-412`) from a `std::array<..., AMREX_SPACEDIM>`, causing out-of-bounds access for 1D/2D MHD builds.

## Severity
`High`

## Affected File
`src/hydro/hydro_system.hpp`

## Affected Function / Symbol
`HydroSystem::ComputeMaxSignalSpeed(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:375`
- Finding tags: portability

## Proposed Patch
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
