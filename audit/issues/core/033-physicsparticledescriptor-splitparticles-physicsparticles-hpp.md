# PhysicsParticleDescriptor<...>::splitParticles(...): no validation that `splitFactor > 0` (`src/particles/PhysicsParticles.hpp:408-469`)

## Summary
no validation that `splitFactor > 0` (`src/particles/PhysicsParticles.hpp:408-469`). `splitFactor == 0` marks old particles for deletion and creates none; negative values can also overflow `max_new_particles` (`:414`) and corrupt ID/resize logic.

## Severity
`High`

## Affected File
`src/particles/PhysicsParticles.hpp`

## Affected Function / Symbol
`PhysicsParticleDescriptor<...>::splitParticles(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:518`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.

## Why This Is a Bug
`splitFactor` controls allocation sizes, ID ranges, loop trip counts, and per-particle mass normalization, but it is never validated. `splitFactor == 0` deletes originals and creates none; negative values produce invalid sizes/offsets and can overflow `max_new_particles`, corrupting particle storage and IDs.

## Complete Code Patch
```diff
diff --git a/src/particles/PhysicsParticles.hpp b/src/particles/PhysicsParticles.hpp
--- a/src/particles/PhysicsParticles.hpp
+++ b/src/particles/PhysicsParticles.hpp
@@
 	void splitParticles(int const lev, int const splitFactor) override
 	{
+		AMREX_ALWAYS_ASSERT_WITH_MESSAGE(splitFactor > 0, "splitParticles requires splitFactor > 0");
 		if (container_ != nullptr && this->getMassIndex() >= 0) {
 			for (typename ContainerType::ParIterType pIter(*container_, lev); pIter.isValid(); ++pIter) {
 				// Update NextID to include particles that will be created
 				const amrex::Long npart_old = pIter.numParticles();
-				const unsigned int max_new_particles = splitFactor * npart_old;
+				const amrex::Long max_new_particles = static_cast<amrex::Long>(splitFactor) * npart_old;
+				AMREX_ALWAYS_ASSERT(max_new_particles >= 0);
 				const amrex::Long pid = ContainerType::ParticleType::NextID();
 				ContainerType::ParticleType::NextID(pid + max_new_particles);
```
