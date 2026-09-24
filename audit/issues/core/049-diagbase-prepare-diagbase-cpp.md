# DiagBase::prepare(...): gated by `if (first_time)` (`src/io/DiagBase.cpp:38`) but never sets `first_time = false`

## Summary
gated by `if (first_time)` (`src/io/DiagBase.cpp:38`) but never sets `first_time = false`. As a result, the one-time filter-device setup path re-runs on every `prepare()` call (and base-class users like `DiagPlotfile`/`DiagParticleTxt` rely on this method directly).

## Severity
`Medium`

## Affected File
`src/io/DiagBase.cpp`

## Affected Function / Symbol
`DiagBase::prepare(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:643`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
