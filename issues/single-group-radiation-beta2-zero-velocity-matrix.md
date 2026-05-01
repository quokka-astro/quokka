# Single-group radiation beta2 flux solve drops the velocity-dependent matrix terms

Severity: High

## Bug

`RadSystem::AddSourceTerms` in `src/radiation/source_terms_single_group.hpp` declares `std::array<double, 3> gasVel{}` inside the nonzero beta-order flux update, but never fills it from the gas momentum and density. In the `beta_order_ != 1` branch with `kappaF != kappaE`, the 3x3 solve is documented as using `A[i][j] = delta_ij * X0 + K0 * v_i * v_j`, but every `gasVel[...]` entry remains zero.

As a result, all velocity outer-product terms vanish. The code silently reduces the beta^2 solve to the simpler diagonal update even when the configuration selected the higher-order terms. Moving radiation and unequal flux/energy opacities therefore get systematically wrong radiation fluxes, gas momentum exchange, and work terms.

## Patch

```diff
diff --git a/src/radiation/source_terms_single_group.hpp b/src/radiation/source_terms_single_group.hpp
--- a/src/radiation/source_terms_single_group.hpp
+++ b/src/radiation/source_terms_single_group.hpp
@@
-				std::array<double, 3> gasVel{};
+				std::array<double, 3> gasVel{};
 				std::array<double, 3> v_terms{};
+				for (int n = 0; n < 3; ++n) {
+					gasVel[n] = gasMtm0[n] / rho;
+				}
 
 				auto fx = Frad_t0[0] / (c_light_ * erad);
 				auto fy = Frad_t0[1] / (c_light_ * erad);
 				auto fz = Frad_t0[2] / (c_light_ * erad);
```
