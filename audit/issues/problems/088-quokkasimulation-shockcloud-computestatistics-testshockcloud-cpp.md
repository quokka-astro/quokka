# QuokkaSimulation<ShockCloud>::ComputeStatistics(): cloud-fraction statistics recompute `C_frac = rho_cloud / (rho_cloud + rho_bg)` without guarding the denominator (`src/problems/ShockCloud/testShockCloud.cpp:636`, `:645`), allowing `nan` statistics for zero partial-density states

## Summary
cloud-fraction statistics recompute `C_frac = rho_cloud / (rho_cloud + rho_bg)` without guarding the denominator (`src/problems/ShockCloud/testShockCloud.cpp:636`, `:645`), allowing `nan` statistics for zero partial-density states.

## Severity
`Medium`

## Affected File
`src/problems/ShockCloud/testShockCloud.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShockCloud>::ComputeStatistics()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1165`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
