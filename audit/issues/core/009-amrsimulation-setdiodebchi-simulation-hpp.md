# AMRSimulation::setDiodeBCHi<...>(...): same issue on the upper boundary; only core hydro fields are filled (`src/simulation.hpp:2753-2764`, `:2800-2805`), so passive scalars/other extra conserved components are not boundary-populated

## Summary
same issue on the upper boundary; only core hydro fields are filled (`src/simulation.hpp:2753-2764`, `:2800-2805`), so passive scalars/other extra conserved components are not boundary-populated.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::setDiodeBCHi<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:252`
- Finding tags: correctness

## Proposed Patch
- After populating the core hydro fields, explicitly copy/fill all remaining conserved components (passive scalars and optional extras) in the diode ghost-fill path.
