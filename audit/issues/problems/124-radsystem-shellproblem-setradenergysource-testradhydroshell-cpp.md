# RadSystem<ShellProblem>::SetRadEnergySource(...): source kernel unconditionally uses z-dimension geometry (`prob_lo[2]`, `dx[2]`, `prob_hi[2]`) (`src/problems/RadhydroShell/testRadhydroShell.cpp:102`, `:114`) without compile-time dimension guards

## Summary
source kernel unconditionally uses z-dimension geometry (`prob_lo[2]`, `dx[2]`, `prob_hi[2]`) (`src/problems/RadhydroShell/testRadhydroShell.cpp:102`, `:114`) without compile-time dimension guards. `problem_main()` is 3D-gated, but this specialization itself is not dimension-safe for 1D/2D builds.

## Severity
`Low`

## Affected File
`src/problems/RadhydroShell/testRadhydroShell.cpp`

## Affected Function / Symbol
`RadSystem<ShellProblem>::SetRadEnergySource(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1712`
- Finding tags: portability

## Proposed Patch
- Make dimensional assumptions explicit: either add `static_assert(AMREX_SPACEDIM == 3)` for intentionally 3D code, or rewrite loops/indexing to use `AMREX_SPACEDIM` with dimension-guarded branches.
