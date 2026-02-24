# QuokkaSimulation<CollapseProblem>::createInitialCICParticles(): computes `particle_mass = total_particle_mass / num_particles` (`src/problems/SphericalCollapse/testSphericalCollapse.cpp:104`) with no guard for `num_particles <= 0`

## Summary
computes `particle_mass = total_particle_mass / num_particles` (`src/problems/SphericalCollapse/testSphericalCollapse.cpp:104`) with no guard for `num_particles <= 0`.

## Severity
`Medium`

## Affected File
`src/problems/SphericalCollapse/testSphericalCollapse.cpp`

## Affected Function / Symbol
`QuokkaSimulation<CollapseProblem>::createInitialCICParticles()`

## Audit Metadata
- Source log: `audit/src-audit-log.md:1817`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
