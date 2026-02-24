# AMRSimulation::readCheckpointHeader(...): parses `istep`, `dt_`, and `tNew_` lines with unbounded `while (lis >> word) { arr[i++] = ...; }` loops (`src/simulation.hpp:4437-4440`, `:4447-4450`, `:4457-4460`) into fixed-size arrays, so malformed headers with extra tokens can overflow

## Summary
parses `istep`, `dt_`, and `tNew_` lines with unbounded `while (lis >> word) { arr[i++] = ...; }` loops (`src/simulation.hpp:4437-4440`, `:4447-4450`, `:4457-4460`) into fixed-size arrays, so malformed headers with extra tokens can overflow.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::readCheckpointHeader(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:294`
- Finding tags: robustness

## Proposed Patch
- Bound the parser loop by the destination array size, validate the exact token count, and reject malformed checkpoint headers instead of continuing to parse.
