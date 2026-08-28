# RadSystem::amendRadState(...): comment says NaN `E_r` is handled, but the floor check uses only `if (E_r < Erad_floor_)` (`src/radiation/radiation_system.hpp:695-701`), which is false for NaN

## Summary
comment says NaN `E_r` is handled, but the floor check uses only `if (E_r < Erad_floor_)` (`src/radiation/radiation_system.hpp:695-701`), which is false for NaN. NaN radiation states can therefore survive `amendRadState()` and trip the subsequent assertion in `PredictStep()` (`src/radiation/radiation_system.hpp:756-760`).

## Severity
`Medium`

## Affected File
`src/radiation/radiation_system.hpp`

## Affected Function / Symbol
`RadSystem::amendRadState(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:404`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
