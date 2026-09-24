# RadSystem::ComputeDustTemperatureBateKeto(...): warm-start branch computes `T_d = T_gas - R_sum / (N_d * sqrt(T_gas))` with no guard on `N_d > 0` and `T_gas > 0` (`src/radiation/radiation_system.hpp:1480-1483`), so low-density/zero-temperature states can generate `inf`/`nan`

## Summary
warm-start branch computes `T_d = T_gas - R_sum / (N_d * sqrt(T_gas))` with no guard on `N_d > 0` and `T_gas > 0` (`src/radiation/radiation_system.hpp:1480-1483`), so low-density/zero-temperature states can generate `inf`/`nan`.

## Severity
`Medium`

## Affected File
`src/radiation/radiation_system.hpp`

## Affected Function / Symbol
`RadSystem::ComputeDustTemperatureBateKeto(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:439`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
