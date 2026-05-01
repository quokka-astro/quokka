# Particle AMR subcycling runs through unsupported code paths

Severity: High

## Explanation

Particle drift/kick code in `src/particles/PhysicsParticles.hpp` documents that AMR subcycling is not supported, and `src/particles/particle_destruction.hpp` has a TODO explaining a concrete failure mode: a particle redistributed to a lower level during a subcycle can miss the second drift step.

`amr.do_subcycle` defaults to `1`, but `src/simulation.hpp` only aborts on subcycling for self-gravity. Particle simulations without self-gravity can therefore run AMR subcycling silently even though the particle algorithms are not designed for it. That can leave particles at inconsistent positions and levels and can make particle-mesh source terms act at the wrong time or level.

## Patch

Reject unsupported particle subcycling during initialization, allowing single-level runs and explicit `do_subcycle = 0` AMR particle runs.

```diff
diff --git a/src/simulation.hpp b/src/simulation.hpp
--- a/src/simulation.hpp
+++ b/src/simulation.hpp
@@
 	int nlevs_max = max_level + 1;
 	istep.resize(nlevs_max, 0);
 	nsubsteps.resize(nlevs_max, 1);
+#if AMREX_SPACEDIM == 3
+	if constexpr (Particle_Traits<problem_t>::particle_switch != ParticleSwitch::None) {
+		if ((do_subcycle == 1) && (max_level > 0)) {
+			amrex::Abort("Particle simulations do not support AMR subcycling. Set amr.do_subcycle = 0.");
+		}
+	}
+#endif
 	if (do_subcycle == 1) {
 		for (int lev = 1; lev <= max_level; ++lev) {
 			nsubsteps[lev] = MaxRefRatio(lev - 1);
```
