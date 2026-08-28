# AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...): species-2 partial density at both boundaries is computed without multiplying the full mass fraction by density (`src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp:155`, `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp:172`)

## Summary
species-2 partial density at both boundaries is computed without multiplying the full mass fraction by density (`src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp:155`, `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp:172`). The expression effectively applies `*rho` only to the sinusoidal term, yielding incorrect (and for the right boundary, dramatically too large) scalar partial density.

## Severity
`Medium`

## Affected File
`src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp`

## Affected Function / Symbol
`AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:934`
- Finding tags: none

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
