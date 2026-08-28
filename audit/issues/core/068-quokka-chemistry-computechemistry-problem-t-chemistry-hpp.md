# quokka::chemistry::computeChemistry<problem_t>(...): the kernel divides species partial densities by `rho` (`src/chemistry/Chemistry.hpp:59-62`) and computes derived fractions (`src/chemistry/Chemistry.hpp:70-72`) before the low-density early-return guard (`src/chemistry/Chemistry.hpp:75-78`)

## Summary
the kernel divides species partial densities by `rho` (`src/chemistry/Chemistry.hpp:59-62`) and computes derived fractions (`src/chemistry/Chemistry.hpp:70-72`) before the low-density early-return guard (`src/chemistry/Chemistry.hpp:75-78`). If `rho <= 0` (or extremely small), this can generate invalid values before the intended density cutoff check.

## Severity
`Medium`

## Affected File
`src/chemistry/Chemistry.hpp`

## Affected Function / Symbol
`quokka::chemistry::computeChemistry<problem_t>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:884`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
