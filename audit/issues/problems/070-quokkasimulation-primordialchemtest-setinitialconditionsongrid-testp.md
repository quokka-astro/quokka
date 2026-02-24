# QuokkaSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(...): if configured species number densities sum to zero, `rhotot` remains zero and normalization divides by `rhotot` (`src/problems/PrimordialChem/testPrimordialChem.cpp:197-211`), producing invalid initial mass fractions/number densities before EOS call

## Summary
if configured species number densities sum to zero, `rhotot` remains zero and normalization divides by `rhotot` (`src/problems/PrimordialChem/testPrimordialChem.cpp:197-211`), producing invalid initial mass fractions/number densities before EOS call.

## Severity
`Medium`

## Affected File
`src/problems/PrimordialChem/testPrimordialChem.cpp`

## Affected Function / Symbol
`QuokkaSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:908`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
