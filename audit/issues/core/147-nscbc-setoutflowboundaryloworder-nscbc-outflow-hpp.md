# NSCBC::setOutflowBoundaryLowOrder<...>(...): in the `DIR == FluxDir::X3` reflecting fallback branch, `Q_im3` is assigned three times and `Q_im4`/`Q_im5` are never populated (`src/hydro/NSCBC_outflow.hpp:427-432`; repeated assignments at `:430-432`)

## Summary
in the `DIR == FluxDir::X3` reflecting fallback branch, `Q_im3` is assigned three times and `Q_im4`/`Q_im5` are never populated (`src/hydro/NSCBC_outflow.hpp:427-432`; repeated assignments at `:430-432`). This corrupts reflected ghost-state construction for z-boundaries in low-order mode.

## Severity
`High`

## Affected File
`src/hydro/NSCBC_outflow.hpp`

## Affected Function / Symbol
`NSCBC::setOutflowBoundaryLowOrder<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1956`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
