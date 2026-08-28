# QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(...): species normalization divides by `rhotot` and then by `msum` (`src/problems/PopIII/testPopIII.cpp:277`, `:282`) with no guard; zero species abundances or `numdens_init == 0` will produce NaNs

## Summary
species normalization divides by `rhotot` and then by `msum` (`src/problems/PopIII/testPopIII.cpp:277`, `:282`) with no guard; zero species abundances or `numdens_init == 0` will produce NaNs.

## Severity
`Medium`

## Affected File
`src/problems/PopIII/testPopIII.cpp`

## Affected Function / Symbol
`QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1862`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
