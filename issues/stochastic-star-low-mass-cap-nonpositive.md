# Nonpositive low-mass composite cap corrupts stochastic star creation

Severity: High

## Explanation

`particles.low_mass_composite_max_mass` is parsed without validation in `src/particles/particle_types.hpp`.

The stochastic star `ParticleChecker` treats a nonpositive cap as "do not split" and returns one low-mass composite particle. The matching `ParticleCreator`, however, unconditionally computes:

```cpp
const int num_low = static_cast<int>(std::ceil(mass_low_mass_star / low_mass_composite_max_mass_));
```

If the input cap is `0` or negative, the checker and creator disagree. The creator can produce `num_low == 0`, a negative value, or an implementation-defined conversion from infinity. That makes `num_high = num_particles - num_low` inconsistent with the allocated particle count and can divide by zero in `mass_low_each`. The result is invalid particle mass/velocity state and possible out-of-bounds particle initialization.

## Patch

Validate the runtime parameter and keep the creator logic identical to the checker logic.

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
 					const int num_high = num_particles - num_low;
 					const amrex::Real mass_low_each = mass_low_mass_star / static_cast<amrex::Real>(num_low);
diff --git a/src/particles/particle_types.hpp b/src/particles/particle_types.hpp
--- a/src/particles/particle_types.hpp
+++ b/src/particles/particle_types.hpp
@@
 	// Low-mass composite particle mass cap (split into multiple particles if exceeded)
 	pp.query("low_mass_composite_max_mass", low_mass_composite_max_mass);
+	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(low_mass_composite_max_mass > 0.0, "particles.low_mass_composite_max_mass must be positive.");
```
