# quokka::turbulence::calculate_dispersion<problem_t>(...): particle-free/zero-mass states are not guarded; reductions divide by `total_rho` (`src/turbulence/TurbulentDriving.hpp:143-150`) and can produce `nan`/`inf`

## Summary
particle-free/zero-mass states are not guarded; reductions divide by `total_rho` (`src/turbulence/TurbulentDriving.hpp:143-150`) and can produce `nan`/`inf`.

## Severity
`Medium`

## Affected File
`src/turbulence/TurbulentDriving.hpp`

## Affected Function / Symbol
`quokka::turbulence::calculate_dispersion<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:543`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
