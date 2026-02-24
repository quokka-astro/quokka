# RadSystem::AddSourceTermsSingleGroup(...): in the `beta_order_ > 1` branch, `gasVel` is value-initialized but never populated before building the 3x3 flux-update matrix (`src/radiation/source_terms_single_group.hpp:399`, `:453-463`)

## Summary
in the `beta_order_ > 1` branch, `gasVel` is value-initialized but never populated before building the 3x3 flux-update matrix (`src/radiation/source_terms_single_group.hpp:399`, `:453-463`). This silently drops the intended velocity-coupling terms in that branch.

## Severity
`Medium`

## Affected File
`src/radiation/source_terms_single_group.hpp`

## Affected Function / Symbol
`RadSystem::AddSourceTermsSingleGroup(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:450`
- Finding tags: none

## Proposed Patch
- Populate `gasVel` from the current gas state before assembling the flux-update matrix in the `beta_order_ > 1` branch, and add a test that exercises that code path.
