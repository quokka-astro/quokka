# read_dataset(...): reads `ndims` but unconditionally indexes `dims[0..2]` when constructing the 3D table (`src/turbulence/TurbDataReader.cpp:27-30`, `:49`)

## Summary
reads `ndims` but unconditionally indexes `dims[0..2]` when constructing the 3D table (`src/turbulence/TurbDataReader.cpp:27-30`, `:49`). Malformed/non-3D datasets can trigger out-of-bounds access.

## Severity
`High`

## Affected File
`src/turbulence/TurbDataReader.cpp`

## Affected Function / Symbol
`read_dataset(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:533`
- Finding tags: robustness

## Proposed Patch
- Replace fixed indices/loop bounds with container-size-aware logic (`AMREX_SPACEDIM` / `.size()`), and add assertions in debug builds to catch future regressions.
