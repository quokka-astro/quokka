# quokka::DataTableGpuConst<Ndim,Nout,oob_policy>::find_interpolation_data(...): assumes at least 2 points per dimension by forcing `indices = sizes-2` at the upper edge (`src/util/DataTable.hpp:123-126`) and dividing by `dcoord[dim]` (`src/util/DataTable.hpp:130-131`)

## Summary
assumes at least 2 points per dimension by forcing `indices = sizes-2` at the upper edge (`src/util/DataTable.hpp:123-126`) and dividing by `dcoord[dim]` (`src/util/DataTable.hpp:130-131`). If any dimension size is 1, this produces invalid indices/divide-by-zero.

## Severity
`Medium`

## Affected File
`src/util/DataTable.hpp`

## Affected Function / Symbol
`quokka::DataTableGpuConst<Ndim,Nout,oob_policy>::find_interpolation_data(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:794`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
