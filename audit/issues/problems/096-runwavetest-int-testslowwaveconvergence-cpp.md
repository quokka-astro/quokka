# runWaveTest(int): computes slow magnetosonic speed `cs` (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:444`) and then `max_time = wavelength / cs` (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:463`) without guarding `cs > 0`

## Summary
computes slow magnetosonic speed `cs` (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:444`) and then `max_time = wavelength / cs` (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:463`) without guarding `cs > 0`. Perpendicular configurations can make `cs` approach or equal zero, causing division by zero/very large stop times.

## Severity
`Medium`

## Affected File
`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp`

## Affected Function / Symbol
`runWaveTest(int)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1285`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
