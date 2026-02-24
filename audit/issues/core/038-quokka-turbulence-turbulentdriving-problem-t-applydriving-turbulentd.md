# quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...): computes and stores `updated` (`src/turbulence/TurbulentDriving.hpp:58`) but always returns `true` (`:98`)

## Summary
computes and stores `updated` (`src/turbulence/TurbulentDriving.hpp:58`) but always returns `true` (`:98`). The return value does not reflect whether the driving field was actually updated/applied.

## Severity
`High`

## Affected File
`src/turbulence/TurbulentDriving.hpp`

## Affected Function / Symbol
`quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:541`
- Finding tags: correctness/API

## Proposed Patch
- Return the actual `updated` flag (or equivalent apply-result) so callers can distinguish no-op timesteps from applied turbulence forcing.
