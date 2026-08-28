# AMRSimulation::evolve(): time-based checkpoint scheduling is initialized with `next_chk_file_time = 0` and not advanced to `checkpointTimeInterval_` on fresh starts (`src/simulation.hpp:1338-1345`), unlike plotfiles (`src/simulation.hpp:1328-1337`)

## Summary
time-based checkpoint scheduling is initialized with `next_chk_file_time = 0` and not advanced to `checkpointTimeInterval_` on fresh starts (`src/simulation.hpp:1338-1345`), unlike plotfiles (`src/simulation.hpp:1328-1337`). This causes `checkpointtime_interval` to trigger on the first completed step (`src/simulation.hpp:1530-1535`) rather than after one full interval.

## Severity
`Medium`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::evolve()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:227`
- Finding tags: none

## Proposed Patch
- Mirror the plotfile scheduling logic on fresh starts by initializing `next_chk_file_time` to the first checkpoint interval (or `cur_time + checkpointTimeInterval_`), then advancing monotonically after writes.
