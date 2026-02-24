# particle_io::saveParticleDataToTxtFile<ContainerType>(...): gathers `particle_ids` (`src/particles/particle_IO.hpp:376`) but never writes them, and integer-component output starts at index `1` (`src/particles/particle_IO.hpp:398`) instead of `0`

## Summary
gathers `particle_ids` (`src/particles/particle_IO.hpp:376`) but never writes them, and integer-component output starts at index `1` (`src/particles/particle_IO.hpp:398`) instead of `0`. This silently drops the first user integer component (and all integer data when `NInt == 1`).

## Severity
`High`

## Affected File
`src/particles/particle_IO.hpp`

## Affected Function / Symbol
`particle_io::saveParticleDataToTxtFile<ContainerType>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:562`
- Finding tags: correctness/output

## Proposed Patch
- Write the gathered particle IDs to the output stream and start integer-component serialization at index `0` so the first user integer component is preserved.
