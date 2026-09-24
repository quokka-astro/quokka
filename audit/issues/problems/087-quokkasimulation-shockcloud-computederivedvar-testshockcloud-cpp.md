# QuokkaSimulation<ShockCloud>::ComputeDerivedVar(...): `cloud_fraction` branch computes `rho_cloud / (rho_cloud + rho_bg)` without a zero/positivity guard (`src/problems/ShockCloud/testShockCloud.cpp:425`)

## Summary
`cloud_fraction` branch computes `rho_cloud / (rho_cloud + rho_bg)` without a zero/positivity guard (`src/problems/ShockCloud/testShockCloud.cpp:425`). If both partial densities vanish (e.g., pathological/ghost-cell state), this yields `nan`.

## Severity
`Medium`

## Affected File
`src/problems/ShockCloud/testShockCloud.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShockCloud>::ComputeDerivedVar(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1162`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
