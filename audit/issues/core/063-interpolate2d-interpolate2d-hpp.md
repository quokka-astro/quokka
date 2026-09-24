# interpolate2d(...): boundary-degenerate branch conditions mistakenly compare coordinate `yi` (double) instead of index `iy` (int) (`src/math/Interpolate2D.hpp:58`, `src/math/Interpolate2D.hpp:63`)

## Summary
boundary-degenerate branch conditions mistakenly compare coordinate `yi` (double) instead of index `iy` (int) (`src/math/Interpolate2D.hpp:58`, `src/math/Interpolate2D.hpp:63`). This misroutes edge-case interpolation logic near table boundaries.

## Severity
`High`

## Affected File
`src/math/Interpolate2D.hpp`

## Affected Function / Symbol
`interpolate2d(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:827`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
