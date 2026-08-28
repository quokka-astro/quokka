# RadSystem<MarshakProblem>::SetRadEnergySource(...): source-cell overlap coordinates are computed as `xl = i*dx` / `xr = (i+1)*dx` (`src/problems/RadSuOlson/testRadSuOlson.cpp:138-139`) while the `prob_lo` argument is ignored (`:134`)

## Summary
source-cell overlap coordinates are computed as `xl = i*dx` / `xr = (i+1)*dx` (`src/problems/RadSuOlson/testRadSuOlson.cpp:138-139`) while the `prob_lo` argument is ignored (`:134`). If the domain lower bound is nonzero, the source region `[0, x0]` is shifted incorrectly.

## Severity
`High`

## Affected File
`src/problems/RadSuOlson/testRadSuOlson.cpp`

## Affected Function / Symbol
`RadSystem<MarshakProblem>::SetRadEnergySource(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1666`
- Finding tags: domain-origin correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
