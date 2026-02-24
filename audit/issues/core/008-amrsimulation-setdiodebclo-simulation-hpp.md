# AMRSimulation::setDiodeBCLo<...>(...): diode ghost fill copies/reflects only `{rho, mom, E, Eint}` and leaves additional conserved components (e.g

## Summary
diode ghost fill copies/reflects only `{rho, mom, E, Eint}` and leaves additional conserved components (e.g. passive scalars) untouched (`src/simulation.hpp:2642-2653`, `:2689-2694`). This can leave stale ghost values when diode BCs are used with extra hydro state components.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::setDiodeBCLo<...>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:251`
- Finding tags: correctness

## Proposed Patch
- After populating the core hydro fields, explicitly copy/fill all remaining conserved components (passive scalars and optional extras) in the diode ghost-fill path.
