# quokka::particleParmParse(): `particle_param3` is declared (`src/particles/particle_types.hpp:471`) but not parsed from inputs; the parser only reads `param1` and `param2` (`src/particles/particle_types.hpp:542-543`), so `particles.param3` is silently ignored

## Summary
`particle_param3` is declared (`src/particles/particle_types.hpp:471`) but not parsed from inputs; the parser only reads `param1` and `param2` (`src/particles/particle_types.hpp:542-543`), so `particles.param3` is silently ignored.

## Severity
`Medium`

## Affected File
`src/particles/particle_types.hpp`

## Affected Function / Symbol
`quokka::particleParmParse()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:479`
- Finding tags: none

## Proposed Patch
- Add the missing `ParmParse` query for the parameter and include a unit/regression test that verifies the runtime value is propagated from inputs.
