# AMRSimulation::RemakeLevel(...): repeats the same face-BC packing bug as `MakeNewLevelFromCoarse` (`BCs_array[idim] = BCs_fc_` at `src/simulation.hpp:2416`), passing flattened BC records where per-direction BC vectors are expected

## Summary
repeats the same face-BC packing bug as `MakeNewLevelFromCoarse` (`BCs_array[idim] = BCs_fc_` at `src/simulation.hpp:2416`), passing flattened BC records where per-direction BC vectors are expected.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::RemakeLevel(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:242`
- Finding tags: correctness

## Proposed Patch
- Slice `BCs_fc_` into per-direction vectors of length `ncomp_per_dim_fc` before calling face-array fill helpers, instead of reusing the full flattened vector for every direction.
