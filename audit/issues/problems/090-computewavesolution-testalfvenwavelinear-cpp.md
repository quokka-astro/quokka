# computeWaveSolution(...): unconditionally accesses `prob_lo[2]` and `dx[2]` (`src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp:221`, `:226`, and FC stencil terms `:277-286`) without a 3D-only guard

## Summary
unconditionally accesses `prob_lo[2]` and `dx[2]` (`src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp:221`, `:226`, and FC stencil terms `:277-286`) without a 3D-only guard. In non-3D builds this helper is not dimension-safe.

## Severity
`Low`

## Affected File
`src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp`

## Affected Function / Symbol
`computeWaveSolution(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1240`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
