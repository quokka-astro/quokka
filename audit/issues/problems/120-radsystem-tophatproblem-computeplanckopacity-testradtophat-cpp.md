# RadSystem<TophatProblem>::ComputePlanckOpacity(...): the fallback branch uses `AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true, "opacity not defined!")` (`src/problems/RadTophat/testRadTophat.cpp:80`), which never fails; unsupported densities silently continue with `kappa == 0`

## Summary
the fallback branch uses `AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true, "opacity not defined!")` (`src/problems/RadTophat/testRadTophat.cpp:80`), which never fails; unsupported densities silently continue with `kappa == 0`.

## Severity
`High`

## Affected File
`src/problems/RadTophat/testRadTophat.cpp`

## Affected Function / Symbol
`RadSystem<TophatProblem>::ComputePlanckOpacity(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1630`
- Finding tags: correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
