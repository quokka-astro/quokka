# ODE error norm uses signed state values in the tolerance scale

Severity: High

## Explanation

`src/math/ODEIntegrate.hpp` computes the adaptive ODE error weight as:

```cpp
Real w_i = 1. / (reltol * y0[i] + abstol[i]);
```

Standard weighted RMS error norms use `abstol + reltol * abs(y_i)`. Using the signed state value can make the denominator too small, negative, or zero for negative-valued ODE components. The squared weight hides the sign, so the integrator can drastically overestimate the error, shrink the timestep unnecessarily, or hit infinities and fail even when the local error is acceptable.

This is generic ODE infrastructure, so any future cooling/chemistry/source integration with a valid negative component is exposed.

## Patch

```diff
diff --git a/src/math/ODEIntegrate.hpp b/src/math/ODEIntegrate.hpp
--- a/src/math/ODEIntegrate.hpp
+++ b/src/math/ODEIntegrate.hpp
@@
 	Real err_sq = 0;
 	for (int i = 0; i < N; ++i) {
-		Real w_i = 1. / (reltol * y0[i] + abstol[i]);
+		const Real scale_i = reltol * std::abs(y0[i]) + abstol[i];
+		AMREX_ASSERT(scale_i > 0.0);
+		Real w_i = 1. / scale_i;
 		err_sq += (yerr[i] * yerr[i]) * (w_i * w_i);
 	}
```
