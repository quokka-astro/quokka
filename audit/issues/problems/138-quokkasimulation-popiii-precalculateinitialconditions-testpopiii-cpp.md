# QuokkaSimulation<PopIII>::preCalculateInitialConditions(): `rms_dv_target` is initialized to `NAN`, queried from `perturb.rms_velocity`, and used in `rescale_factor = rms_dv_target / rms_dv_actual` without validation (`src/problems/PopIII/testPopIII.cpp:161-165`); missing input silently seeds NaNs into the IC velocity field

## Summary
`rms_dv_target` is initialized to `NAN`, queried from `perturb.rms_velocity`, and used in `rescale_factor = rms_dv_target / rms_dv_actual` without validation (`src/problems/PopIII/testPopIII.cpp:161-165`); missing input silently seeds NaNs into the IC velocity field.

## Severity
`Medium`

## Affected File
`src/problems/PopIII/testPopIII.cpp`

## Affected Function / Symbol
`QuokkaSimulation<PopIII>::preCalculateInitialConditions()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1860`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
