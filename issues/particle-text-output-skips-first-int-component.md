# Particle text output omits the first integer component

Severity: High

## Explanation

`particle_io::saveParticleDataToTxtFile` advertises output of particle positions, real components, and integer components. The integer-component loop is guarded with `NInt > 1` and starts at index `1`:

```cpp
if constexpr (ContainerType::ParticleType::NInt > 1) {
	for (size_t j = 1; j < int_data[i].size(); ++j) {
		outFile << int_data[i][j] << " ";
	}
}
```

This drops all integer data for particle types with exactly one integer component, including the stellar evolution stage for `StochasticStellarPop` and `Test` particles. If a type ever has multiple integer components, component 0 is still lost. The resulting text diagnostics cannot reconstruct particle state or distinguish stellar stages.

## Patch

```diff
diff --git a/src/particles/particle_IO.hpp b/src/particles/particle_IO.hpp
--- a/src/particles/particle_IO.hpp
+++ b/src/particles/particle_IO.hpp
@@
-			if constexpr (ContainerType::ParticleType::NInt > 1) {
-				for (size_t j = 1; j < int_data[i].size(); ++j) {
+			if constexpr (ContainerType::ParticleType::NInt > 0) {
+				for (size_t j = 0; j < int_data[i].size(); ++j) {
 					outFile << int_data[i][j] << " ";
 				}
 			}
```
