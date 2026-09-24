# MHDSystem::ComputeEMF_FelkerStone2017(...): hard-codes 3D indexing (`fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:179-186` and `:338-340`, so it is not safe for 1D/2D MHD builds

## Summary
hard-codes 3D indexing (`fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:179-186` and `:338-340`, so it is not safe for 1D/2D MHD builds.

## Severity
`Low`

## Affected File
`src/hydro/mhd_system.hpp`

## Affected Function / Symbol
`MHDSystem::ComputeEMF_FelkerStone2017(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:391`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
