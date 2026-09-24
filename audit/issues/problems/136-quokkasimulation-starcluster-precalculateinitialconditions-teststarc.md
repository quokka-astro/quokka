# QuokkaSimulation<StarCluster>::preCalculateInitialConditions(): virial normalization computes `M_sphere ~ R_sphere^3`, `rms_dv_target ~ 

## Summary
virial normalization computes `M_sphere ~ R_sphere^3`, `rms_dv_target ~ ... / R_sphere`, and `rescale_factor = rms_dv_target / rms_dv_actual` without guarding `R_sphere > 0` or `rms_dv_actual > 0` (`src/problems/StarCluster/testStarCluster.cpp:104-108`).

## Severity
`Medium`

## Affected File
`src/problems/StarCluster/testStarCluster.cpp`

## Affected Function / Symbol
`QuokkaSimulation<StarCluster>::preCalculateInitialConditions()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1850`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
