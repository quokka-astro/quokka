# problem_main(): `stop_time` calculation divides by `vortex_u_magn = vortex_Mach * sound_speed` (`src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp:235`, `:264`, `:268`) without validating `vortex_Mach > 0`

## Summary
`stop_time` calculation divides by `vortex_u_magn = vortex_Mach * sound_speed` (`src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp:235`, `:264`, `:268`) without validating `vortex_Mach > 0`. A user input of `setup.vortex_Mach = 0` yields division by zero (`inf`/`nan` stop time).

## Severity
`Medium`

## Affected File
`src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1139`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
