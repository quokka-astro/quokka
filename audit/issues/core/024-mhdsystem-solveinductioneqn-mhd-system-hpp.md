# MHDSystem::SolveInductionEqn(...): loops over `w0 = 0..2` unconditionally (`src/hydro/mhd_system.hpp:957`) while operating on `std::array<..., AMREX_SPACEDIM>` containers

## Summary
loops over `w0 = 0..2` unconditionally (`src/hydro/mhd_system.hpp:957`) while operating on `std::array<..., AMREX_SPACEDIM>` containers. This is out-of-bounds for 1D/2D MHD builds and should use `AMREX_SPACEDIM`.

## Severity
`High`

## Affected File
`src/hydro/mhd_system.hpp`

## Affected Function / Symbol
`MHDSystem::SolveInductionEqn(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:398`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
