# problem_main(): final mass check also uses one-sided relative difference and divides by `m_star_tot2` (`src/problems/ParticleSF/testParticleSF.cpp:325-327`) without `abs(...)` or a zero guard, weakening/falsifying failure detection when stellar mass is small or underpredicted

## Summary
final mass check also uses one-sided relative difference and divides by `m_star_tot2` (`src/problems/ParticleSF/testParticleSF.cpp:325-327`) without `abs(...)` or a zero guard, weakening/falsifying failure detection when stellar mass is small or underpredicted.

## Severity
`Medium`

## Affected File
`src/problems/ParticleSF/testParticleSF.cpp`

## Affected Function / Symbol
`problem_main()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1448`
- Finding tags: test validity/robustness

## Proposed Patch
- Use absolute relative error (`std::abs(...)`) and guard zero denominators in pass/fail checks so underestimates and zero-reference cases are handled correctly.
