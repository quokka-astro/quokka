# QuokkaSimulation::replaceEMFs(...): loops `for (int iedge = 0; iedge < 3; ++iedge)` over `std::array<amrex::MultiFab, AMREX_SPACEDIM>` (`src/QuokkaSimulation.hpp:2507`), which is out-of-bounds for 1D/2D MHD builds

## Summary
loops `for (int iedge = 0; iedge < 3; ++iedge)` over `std::array<amrex::MultiFab, AMREX_SPACEDIM>` (`src/QuokkaSimulation.hpp:2507`), which is out-of-bounds for 1D/2D MHD builds.

## Severity
`High`

## Affected File
`src/QuokkaSimulation.hpp`

## Affected Function / Symbol
`QuokkaSimulation::replaceEMFs(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:333`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
