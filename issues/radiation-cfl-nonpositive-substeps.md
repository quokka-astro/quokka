# Non-positive radiation CFL can produce an invalid subcycle count

Severity: High

## Explanation

`QuokkaSimulation<problem_t>::readParmParse()` accepts `radiation.cfl` without validating that it is finite and positive. `computeNumberOfRadiationSubsteps()` then computes

```cpp
dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat);
nsubSteps = ceil(dt_lev_hydro / dtrad_tmp);
```

If an input file sets `radiation.cfl = 0`, `dtrad_tmp` is zero and the conversion of infinity to `int` is undefined/implementation-defined before the later subcycle assertions can provide a controlled failure. If `radiation.cfl < 0`, `nsubSteps` can be negative and `dt_radiation` can become negative. Either case corrupts timestep control or aborts after an invalid calculation instead of rejecting the configuration at parse time.

This is user-facing because `radiation.cfl` is documented in `docs/markdown/parameters.md` and is read directly from problem inputs.

## Patch

Validate the radiation CFL as soon as it is parsed:

```diff
diff --git a/src/QuokkaSimulation.hpp b/src/QuokkaSimulation.hpp
--- a/src/QuokkaSimulation.hpp
+++ b/src/QuokkaSimulation.hpp
@@
 	if constexpr (Physics_Traits<problem_t>::is_radiation_enabled) {
 		amrex::ParmParse rpp("radiation");
 		rpp.query("cfl", radiationCflNumber_);
+		if (!std::isfinite(radiationCflNumber_) || radiationCflNumber_ <= 0.0) {
+			amrex::Abort("radiation.cfl must be finite and positive.");
+		}
 		rpp.query("max_substeps", maxSubsteps_);
 		rpp.query("print_rad_counter", print_rad_counter_);
 		rpp.query("iteration_tolerance", radiation_iteration_tolerance_);
```
