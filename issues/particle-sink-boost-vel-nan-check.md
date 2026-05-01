# ParticleSink accepts missing `boost_vel_x`

Severity: High

## Explanation

`src/problems/ParticleSink/testParticleSink.cpp` initializes `boost_vel_x` to `NAN`, calls `pp.query("boost_vel_x", boost_vel_x)`, and then checks:

```cpp
AMREX_ASSERT_WITH_MESSAGE(boost_vel_x != NAN, "boost_vel_x must be set in the input file");
```

NaN never compares equal to anything, including itself. Therefore `boost_vel_x != NAN` is always true, so a missing `problem.boost_vel_x` silently leaves `boost_vel_x` as NaN. That value is later assigned into `sim2.userData_.boost_velocity`, allowing the boosted restart comparison to propagate NaNs instead of failing at input validation.

The check is also an `AMREX_ASSERT`, so release builds may compile it away. This is a required runtime input and should be validated unconditionally.

## Patch

```diff
diff --git a/src/problems/ParticleSink/testParticleSink.cpp b/src/problems/ParticleSink/testParticleSink.cpp
--- a/src/problems/ParticleSink/testParticleSink.cpp
+++ b/src/problems/ParticleSink/testParticleSink.cpp
@@
 #include "math/interpolate.hpp"
 #include "util/fextract.hpp"
+#include <cmath>
 #include <format>
 #include <numeric>
 #include <utility>
@@
 	pp.query("particles_file", particles_file);
 	pp.query("refine_half_domain", refine_half_domain);
-	double boost_vel_x = NAN;
-	pp.query("boost_vel_x", boost_vel_x);
-	AMREX_ASSERT_WITH_MESSAGE(boost_vel_x != NAN, "boost_vel_x must be set in the input file");
+	double boost_vel_x = NAN;
+	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(pp.query("boost_vel_x", boost_vel_x) != 0 && std::isfinite(boost_vel_x),
+					 "boost_vel_x must be set to a finite value in the input file");
 
 	// Problem initialization
 	QuokkaSimulation<SinkProblem> sim;
```
