# QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep(): expectation checks use one-sided normalized differences without `abs(...)` (`src/problems/ParticleSF/testParticleSF.cpp:209-223`)

## Summary
expectation checks use one-sided normalized differences without `abs(...)` (`src/problems/ParticleSF/testParticleSF.cpp:209-223`). Large underestimates can still pass because negative relative errors satisfy `< tol`.

## Severity
`Medium`

## Affected File
`src/problems/ParticleSF/testParticleSF.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1447`
- Finding tags: test validity

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
