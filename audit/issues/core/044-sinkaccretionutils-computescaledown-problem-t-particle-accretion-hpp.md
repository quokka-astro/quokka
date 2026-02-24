# SinkAccretionUtils::ComputeScaleDown<problem_t>(...): Jeans-density limiter branch is guarded by `if (accretion_rate_cell > std::numeric_limits<double>::min())` (`src/particles/particle_accretion.hpp:289`), but accretion-zone rates are accumulated as non-positive values (`src/particles/particle_accretion.hpp:237-239`)

## Summary
Jeans-density limiter branch is guarded by `if (accretion_rate_cell > std::numeric_limits<double>::min())` (`src/particles/particle_accretion.hpp:289`), but accretion-zone rates are accumulated as non-positive values (`src/particles/particle_accretion.hpp:237-239`). The intended limiter branch is effectively skipped.

## Severity
`High`

## Affected File
`src/particles/particle_accretion.hpp`

## Affected Function / Symbol
`SinkAccretionUtils::ComputeScaleDown<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:571`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
