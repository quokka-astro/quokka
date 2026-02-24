# QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGrid(...): host `PinnedVector` pointers are extracted via `dataPtr()` (`src/problems/DiskGalaxy/testDiskGalaxy.cpp:205-209`) and captured into an `AMREX_GPU_DEVICE` kernel (`:248`) where they are dereferenced by interpolation lambdas (`:272-325`)

## Summary
host `PinnedVector` pointers are extracted via `dataPtr()` (`src/problems/DiskGalaxy/testDiskGalaxy.cpp:205-209`) and captured into an `AMREX_GPU_DEVICE` kernel (`:248`) where they are dereferenced by interpolation lambdas (`:272-325`). This relies on host-pinned memory being device-accessible and violates the repo’s GPU-lambda safety guidance.

## Severity
`High`

## Affected File
`src/problems/DiskGalaxy/testDiskGalaxy.cpp`

## Affected Function / Symbol
`QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1886`
- Finding tags: GPU portability/safety

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
