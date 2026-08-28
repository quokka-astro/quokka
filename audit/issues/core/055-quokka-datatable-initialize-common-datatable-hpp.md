# quokka::DataTable<...>::initialize_common(...): validates only `sizes_[dim] > 0` (`src/util/DataTable.hpp:721`) but later computes `dcoord_[dim] = 

## Summary
validates only `sizes_[dim] > 0` (`src/util/DataTable.hpp:721`) but later computes `dcoord_[dim] = ... / (sizes_[dim]-1)` (`src/util/DataTable.hpp:763`). `size==1` tables are accepted but create zero-division/invalid interpolation state.

## Severity
`Medium`

## Affected File
`src/util/DataTable.hpp`

## Affected Function / Symbol
`quokka::DataTable<...>::initialize_common(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:797`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
