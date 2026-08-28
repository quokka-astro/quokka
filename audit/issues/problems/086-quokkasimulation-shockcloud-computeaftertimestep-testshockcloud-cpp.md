# QuokkaSimulation<ShockCloud>::computeAfterTimestep(): frame-shift update computes `vx_cm = xmom / cloud_mass` (`src/problems/ShockCloud/testShockCloud.cpp:232`) without guarding `cloud_mass <= 0`

## Summary
frame-shift update computes `vx_cm = xmom / cloud_mass` (`src/problems/ShockCloud/testShockCloud.cpp:232`) without guarding `cloud_mass <= 0`. If the tracked cloud mass becomes zero/underflows, the frame-shift state and subsequent momentum/energy updates become invalid (`inf`/`nan`).

## Severity
`Medium`

## Affected File
`src/problems/ShockCloud/testShockCloud.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ShockCloud>::computeAfterTimestep()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1160`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
