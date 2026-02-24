# MHDSystem::ComputeEMF_Balsara2025(...): hard-codes 3D storage/loops (`cc_mf_EMF(...,3,...)`, `fcx_mf_cVars[2]`, `idim < 3`, `iedge < 3`) at `src/hydro/mhd_system.hpp:501-503`, `:514-523`, `:532`, and `:571`, so the implementation is not dimension-safe for 1D/2D MHD builds

## Summary
hard-codes 3D storage/loops (`cc_mf_EMF(...,3,...)`, `fcx_mf_cVars[2]`, `idim < 3`, `iedge < 3`) at `src/hydro/mhd_system.hpp:501-503`, `:514-523`, `:532`, and `:571`, so the implementation is not dimension-safe for 1D/2D MHD builds.

## Severity
`Medium`

## Affected File
`src/hydro/mhd_system.hpp`

## Affected Function / Symbol
`MHDSystem::ComputeEMF_Balsara2025(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:393`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
