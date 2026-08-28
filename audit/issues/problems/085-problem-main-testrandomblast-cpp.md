# problem_main(): unconditionally extracts a z-axis slice with `fextract(..., 2, ...)` (`src/problems/RandomBlast/testRandomBlast.cpp:203`) without a dimension guard or 3D-only assertion

## Summary
unconditionally extracts a z-axis slice with `fextract(..., 2, ...)` (`src/problems/RandomBlast/testRandomBlast.cpp:203`) without a dimension guard or 3D-only assertion. In 1D/2D builds, this driver is not dimension-safe.

## Severity
`Low`

## Affected File
`src/problems/RandomBlast/testRandomBlast.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1150`
- Finding tags: portability

## Proposed Patch
- Add compile-time dimension guards or refactor the implementation so indexing and stencil accesses are valid for the supported `AMREX_SPACEDIM` values.
