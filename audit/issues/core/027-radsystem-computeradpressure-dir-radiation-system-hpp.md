# RadSystem::ComputeRadPressure<DIR>(...): NaN checks use `AMREX_ASSERT(Fn != NAN)` / `Tn* != NAN` (`src/radiation/radiation_system.hpp:1026-1029`), which are ineffective because comparisons with `NaN` are always true in IEEE arithmetic

## Summary
NaN checks use `AMREX_ASSERT(Fn != NAN)` / `Tn* != NAN` (`src/radiation/radiation_system.hpp:1026-1029`), which are ineffective because comparisons with `NaN` are always true in IEEE arithmetic.

## Severity
`Medium`

## Affected File
`src/radiation/radiation_system.hpp`

## Affected Function / Symbol
`RadSystem::ComputeRadPressure<DIR>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:410`
- Finding tags: robustness

## Proposed Patch
- Replace NaN comparisons with `amrex::Math::isfinite(...)` / `std::isfinite(...)` checks (or `x == x` if required on device) and assert on non-finite values.
