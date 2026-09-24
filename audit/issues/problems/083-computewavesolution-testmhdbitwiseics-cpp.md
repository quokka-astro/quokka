# computeWaveSolution(...): unconditionally reads `prob_lo[2]` and `dx[2]` (`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp:56-58`) even though the file’s explicit 3D requirement is only enforced later inside `verifyPeriodicBCs(...)` (`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp:159-160`)

## Summary
unconditionally reads `prob_lo[2]` and `dx[2]` (`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp:56-58`) even though the file’s explicit 3D requirement is only enforced later inside `verifyPeriodicBCs(...)` (`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp:159-160`). In non-3D builds, initialization/reference paths can hit out-of-bounds access before the guard runs.

## Severity
`High`

## Affected File
`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp`

## Affected Function / Symbol
`computeWaveSolution(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1120`
- Finding tags: portability/robustness

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
