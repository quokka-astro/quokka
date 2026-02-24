# runWaveTest(int): computes `cA = alfven_speed * abs(cos(angle_between_k_b0_rad))` (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:381`) and then `max_time = wavelength / cA` (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:400`) without guarding `cA > 0`

## Summary
computes `cA = alfven_speed * abs(cos(angle_between_k_b0_rad))` (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:381`) and then `max_time = wavelength / cA` (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:400`) without guarding `cA > 0`. A perpendicular setup (`angle_between_k_b0 = 90 deg`) yields division by zero.

## Severity
`Medium`

## Affected File
`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp`

## Affected Function / Symbol
`runWaveTest(int)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1254`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
