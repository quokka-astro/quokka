# QuokkaSimulation<TurbulentBox>::refineGrid(...): AMR tagger unconditionally samples y/z neighbors (`state(..., j±1, ...)`, `state(..., k±1, ...)`) (`src/problems/Turbulence/testTurbulence.cpp:85`, `:87`) and computes a 3D gradient norm, so the implementation is hard-coded for 3D without a dimension guard

## Summary
AMR tagger unconditionally samples y/z neighbors (`state(..., j±1, ...)`, `state(..., k±1, ...)`) (`src/problems/Turbulence/testTurbulence.cpp:85`, `:87`) and computes a 3D gradient norm, so the implementation is hard-coded for 3D without a dimension guard.

## Severity
`Low`

## Affected File
`src/problems/Turbulence/testTurbulence.cpp`

## Affected Function / Symbol
`QuokkaSimulation<TurbulentBox>::refineGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1841`
- Finding tags: portability

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
