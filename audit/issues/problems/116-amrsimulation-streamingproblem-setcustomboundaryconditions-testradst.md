# AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(...): applies `setConstantDirichletBCLo<1>` / `setConstantDirichletBCHi<1>` unconditionally (`src/problems/RadStreamingY/testRadStreamingY.cpp:139-140`), so the specialization is not 1D-safe

## Summary
applies `setConstantDirichletBCLo<1>` / `setConstantDirichletBCHi<1>` unconditionally (`src/problems/RadStreamingY/testRadStreamingY.cpp:139-140`), so the specialization is not 1D-safe.

## Severity
`Low`

## Affected File
`src/problems/RadStreamingY/testRadStreamingY.cpp`

## Affected Function / Symbol
`AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1612`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
