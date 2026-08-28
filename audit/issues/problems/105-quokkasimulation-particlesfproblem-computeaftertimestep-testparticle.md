# QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep(): `mean_mass_high_mass_stars = m_star_high_tot / n_star_high` (`src/problems/ParticleSF/testParticleSF.cpp:153`) can divide by zero if step-1 stochastic sampling produces no high-mass stars

## Summary
`mean_mass_high_mass_stars = m_star_high_tot / n_star_high` (`src/problems/ParticleSF/testParticleSF.cpp:153`) can divide by zero if step-1 stochastic sampling produces no high-mass stars.

## Severity
`Medium`

## Affected File
`src/problems/ParticleSF/testParticleSF.cpp`

## Affected Function / Symbol
`QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1445`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
