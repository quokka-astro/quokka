# AMRSimulation::MakeNewLevelFromCoarse(...): constructs per-direction face BC arrays with `BCs_array[idim] = BCs_fc_` (`src/simulation.hpp:2367`) instead of slicing the flattened `BCs_fc_` vector to `ncomp_per_dim_fc` entries

## Summary
constructs per-direction face BC arrays with `BCs_array[idim] = BCs_fc_` (`src/simulation.hpp:2367`) instead of slicing the flattened `BCs_fc_` vector to `ncomp_per_dim_fc` entries. `FillCoarsePatchFaceArray(...)` then receives incorrect BC vector sizes/ordering.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::MakeNewLevelFromCoarse(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:240`
- Finding tags: correctness

## Proposed Patch
- Slice `BCs_fc_` into per-direction vectors of length `ncomp_per_dim_fc` before calling face-array fill helpers, instead of reusing the full flattened vector for every direction.
