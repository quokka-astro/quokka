# ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleCreator<problem_t>::operator()(...): computes `num_low = ceil(mass_low_mass_star / low_mass_composite_max_mass_)` unconditionally (`src/particles/particle_creation.hpp:499`), but the checker explicitly treats `low_mass_composite_max_mass_ <= 0` as “no splitting” (`src/particles/particle_creation.hpp:444-446`)

## Summary
computes `num_low = ceil(mass_low_mass_star / low_mass_composite_max_mass_)` unconditionally (`src/particles/particle_creation.hpp:499`), but the checker explicitly treats `low_mass_composite_max_mass_ <= 0` as “no splitting” (`src/particles/particle_creation.hpp:444-446`). Non-positive `low_mass_composite_max_mass` can therefore trigger division by zero/invalid `num_low` in the creator path.

## Severity
`High`

## Affected File
`src/particles/particle_creation.hpp`

## Affected Function / Symbol
`ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleCreator<problem_t>::operator()(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:586`
- Finding tags: robustness/correctness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.

## Why This Is a Bug
The checker and creator paths disagree on how `low_mass_composite_max_mass_ <= 0` is handled. The checker treats non-positive values as “no splitting” (`num_low = 1`), but the creator unconditionally divides by `low_mass_composite_max_mass_`, which can divide by zero and produce invalid `num_low`/mass partitioning for exactly the same parameter set.

## Complete Code Patch
```diff
diff --git a/src/particles/particle_creation.hpp b/src/particles/particle_creation.hpp
--- a/src/particles/particle_creation.hpp
+++ b/src/particles/particle_creation.hpp
@@
-					const int num_low = static_cast<int>(std::ceil(mass_low_mass_star / low_mass_composite_max_mass_));
+					int num_low = 1;
+					if ((low_mass_composite_max_mass_ > 0.0) && (mass_low_mass_star > low_mass_composite_max_mass_)) {
+						num_low = static_cast<int>(std::ceil(mass_low_mass_star / low_mass_composite_max_mass_));
+					}
+					AMREX_ASSERT(num_low > 0);
 					const int num_high = num_particles - num_low;
```
