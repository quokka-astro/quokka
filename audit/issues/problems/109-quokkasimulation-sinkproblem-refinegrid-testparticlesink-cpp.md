# QuokkaSimulation<SinkProblem>::refineGrid(...): normalized coordinates are computed as `((i+0.5)*dx)/(phi-plo)` (`src/problems/ParticleSink/testParticleSink.cpp:152-154`) without subtracting `plo`, so the selected refinement subregion shifts if the domain lower bound is nonzero

## Summary
normalized coordinates are computed as `((i+0.5)*dx)/(phi-plo)` (`src/problems/ParticleSink/testParticleSink.cpp:152-154`) without subtracting `plo`, so the selected refinement subregion shifts if the domain lower bound is nonzero.

## Severity
`Medium`

## Affected File
`src/problems/ParticleSink/testParticleSink.cpp`

## Affected Function / Symbol
`QuokkaSimulation<SinkProblem>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1459`
- Finding tags: AMR region selection

## Proposed Patch
- Normalize coordinates relative to the domain lower bound by subtracting `ProbLo()`/`plo` before scaling to `[0,1]` region-selection coordinates.
