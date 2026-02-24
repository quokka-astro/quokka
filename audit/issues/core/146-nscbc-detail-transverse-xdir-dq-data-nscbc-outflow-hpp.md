# NSCBC::detail::transverse_xdir_dQ_data(...): in the `AMREX_SPACEDIM == 3` z-derivative branch, the computed derivative is assigned to `dQ_dy_data` instead of `dQ_dz_data` (`src/hydro/NSCBC_outflow.hpp:132-137`, assignment at `:136`)

## Summary
in the `AMREX_SPACEDIM == 3` z-derivative branch, the computed derivative is assigned to `dQ_dy_data` instead of `dQ_dz_data` (`src/hydro/NSCBC_outflow.hpp:132-137`, assignment at `:136`). This drops the z-transverse contribution and corrupts 3D x-boundary NSCBC transverse terms.

## Severity
`High`

## Affected File
`src/hydro/NSCBC_outflow.hpp`

## Affected Function / Symbol
`NSCBC::detail::transverse_xdir_dQ_data(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1950`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
