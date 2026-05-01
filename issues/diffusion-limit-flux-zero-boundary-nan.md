# Diffusion-limit radiation flux returns NaN at a zero lower group boundary

Severity: High

## Bug

`RadSystem::ComputeFluxInDiffusionLimit` evaluates

```cpp
x * (std::pow(x, 3) / (std::exp(x) - 1.0))
```

for every radiation group boundary. The default and many multi-group boundary arrays start at `0.0`, so the first boundary uses `x == 0`. That makes the inner ratio `0 / (exp(0) - 1) == 0 / 0`, producing `NaN`; the first group's flux is then computed from `edge_values[1] - edge_values[0]` and becomes `NaN`.

This helper is used by `src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp` to initialize radiation fluxes. A zero lower boundary therefore contaminates initial conditions before the solver starts.

## Patch

```diff
diff --git a/src/radiation/radiation_system.hpp b/src/radiation/radiation_system.hpp
--- a/src/radiation/radiation_system.hpp
+++ b/src/radiation/radiation_system.hpp
@@
 	for (int g = 0; g < nGroups_ + 1; ++g) {
 		auto x = coeff * rad_boundaries[g];
-		edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x) - 1. / 3. * x * (std::pow(x, 3) / (std::exp(x) - 1.0)) / gInf;
+		double correction = 0.0;
+		if (x > 1.0e-10) {
+			correction = x * (std::pow(x, 3) / (std::exp(x) - 1.0)) / gInf;
+		}
+		edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x) - 1. / 3. * correction;
 		// test: reproduce the Planck function
 		// edge_values[g] = 4. / 3. * integrate_planck_from_0_to_x(x);
 	}
```
