# AMRSimulation::RemakeLevel(...): `int_state_old_cc` is allocated but never filled before `std::swap(int_state_old_cc, state_old_cc_[level])` (`src/simulation.hpp:2386-2389`), so `state_old_cc_[level]` becomes uninitialized after remaking a level

## Summary
`int_state_old_cc` is allocated but never filled before `std::swap(int_state_old_cc, state_old_cc_[level])` (`src/simulation.hpp:2386-2389`), so `state_old_cc_[level]` becomes uninitialized after remaking a level.

## Severity
`High`

## Affected File
`src/simulation.hpp`

## Affected Function / Symbol
`AMRSimulation::RemakeLevel(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:241`
- Finding tags: correctness

## Proposed Patch
- Initialize or copy-populate the temporary state before swapping it into the live state vector; add a regression test that exercises the remake/remap path.
