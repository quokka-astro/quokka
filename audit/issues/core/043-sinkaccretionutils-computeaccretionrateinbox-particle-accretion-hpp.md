# SinkAccretionUtils::ComputeAccretionRateInBox<...>(...): relative accretion rate uses `

## Summary
relative accretion rate uses `... / (vol * rho)` (`src/particles/particle_accretion.hpp:235-239`) without a runtime guard on `rho > 0` beyond an `AMREX_ASSERT`, so release builds can emit invalid rates if zero-density cells appear.

## Severity
`Medium`

## Affected File
`src/particles/particle_accretion.hpp`

## Affected Function / Symbol
`SinkAccretionUtils::ComputeAccretionRateInBox<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:570`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
