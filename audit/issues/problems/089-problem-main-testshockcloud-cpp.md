# problem_main(): shock-jump setup divides by `M0` when computing `v_wind` (`src/problems/ShockCloud/testShockCloud.cpp:765`) and uses `v_shock = M0 * x4` downstream (`src/problems/ShockCloud/testShockCloud.cpp:767`, `:781`, `:786`) without validating `Mach_shock > 0`

## Summary
shock-jump setup divides by `M0` when computing `v_wind` (`src/problems/ShockCloud/testShockCloud.cpp:765`) and uses `v_shock = M0 * x4` downstream (`src/problems/ShockCloud/testShockCloud.cpp:767`, `:781`, `:786`) without validating `Mach_shock > 0`. Zero/invalid `Mach_shock` inputs can produce divide-by-zero/`nan` setup values.

## Severity
`Medium`

## Affected File
`src/problems/ShockCloud/testShockCloud.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1168`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
