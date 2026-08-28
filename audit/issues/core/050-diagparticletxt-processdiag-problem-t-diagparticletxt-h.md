# DiagParticleTxt::processDiag<problem_t>(...): header comment/member comment say empty `m_particleTypes` means "all" (`src/io/DiagParticleTxt.H:30`), and `init()` prints "Including all particle types" when none are specified; but `processDiag()` skips output when the list is empty (`src/io/DiagParticleTxt.H:55-60`)

## Summary
header comment/member comment say empty `m_particleTypes` means "all" (`src/io/DiagParticleTxt.H:30`), and `init()` prints "Including all particle types" when none are specified; but `processDiag()` skips output when the list is empty (`src/io/DiagParticleTxt.H:55-60`). Default configuration therefore emits no particle diagnostic.

## Severity
`Medium`

## Affected File
`src/io/DiagParticleTxt.H`

## Affected Function / Symbol
`DiagParticleTxt::processDiag<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:673`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
