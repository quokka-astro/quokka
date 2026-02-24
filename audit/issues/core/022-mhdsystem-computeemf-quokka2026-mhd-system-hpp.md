# MHDSystem::ComputeEMF_Quokka2026(...): hard-codes 3D indexing (`fcx_mf_vel[2]`, `fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:371-385` and `:475-477`, causing out-of-bounds access in 1D/2D MHD builds

## Summary
hard-codes 3D indexing (`fcx_mf_vel[2]`, `fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:371-385` and `:475-477`, causing out-of-bounds access in 1D/2D MHD builds.

## Severity
`High`

## Affected File
`src/hydro/mhd_system.hpp`

## Affected Function / Symbol
`MHDSystem::ComputeEMF_Quokka2026(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:392`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
