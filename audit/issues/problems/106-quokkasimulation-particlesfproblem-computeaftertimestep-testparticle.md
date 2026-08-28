# QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep(): histogram slope diagnostic takes `std::log(hist[0])` / `std::log(hist[n_bins-1])` (`src/problems/ParticleSF/testParticleSF.cpp:187`) without checking empty bins; zero counts yield `-inf`/`nan`

## Summary
histogram slope diagnostic takes `std::log(hist[0])` / `std::log(hist[n_bins-1])` (`src/problems/ParticleSF/testParticleSF.cpp:187`) without checking empty bins; zero counts yield `-inf`/`nan`.

## Severity
`Medium`

## Affected File
`src/problems/ParticleSF/testParticleSF.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1446`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
