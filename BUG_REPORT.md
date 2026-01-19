# BUG_REPORT

## Title
Restart refinement + particle headers cause invalid particles and abort in BinaryOrbitCICRefactor2

## Summary
When `restartRefineFactor_ > 1` and the checkpoint contains `Particle_H`, AMReX takes the dual-grid restart path. The particle box arrays in `Particle_H` correspond to the *coarse* checkpoint grid, but the restart refines the simulation grid. This mismatch makes particles invalid for the refined grid, leading to an abort in `ParticleContainer::locateParticle()` during restart.

## Symptoms
- Test: `BinaryOrbitCICRefactor2`
- Error: `amrex::Abort::0::ParticleContainer::locateParticle(): invalid particle. !!!`
- Occurs immediately after restart when `restartRefineFactor_ = 2` and `Particle_H` exists in the checkpoint.

## Reproduction
1. Build with AMReX particles enabled.
2. Run:
   - `ctest -R "BinaryOrbitCICRefactorInit2" -VV --output-on-failure`
   - `ctest -R "BinaryOrbitCICRefactor2" -VV --output-on-failure`
3. Observe abort in restart for `BinaryOrbitCICRefactor2`.

## Root Cause
The restart refinement path changes the grid resolution, but the particle restart code assumes particle box arrays in `Particle_H` are still compatible with the current grid. With `restartRefineFactor_ > 1`, those box arrays are *coarse* while the simulation grid is *refined*. AMReX then tries to locate particles using incompatible box arrays and raises an invalid-particle abort.

### Source references
- Restart refinement context and grid refinement on restart:
  - `src/simulation.hpp` (`ReadCheckpointFile`, refinement context, `restartRefineFactor_`)
- Particle restart logic (dual-grid path):
  - `extern/amrex/Src/Particle/AMReX_ParticleIO.H` (`ParticleContainer_impl::Restart`)
- Abort site:
  - `extern/amrex/Src/Particle/AMReX_ParticleLocator.H` (`locateParticle`) via restart call stack

## Evidence
- Checkpoint header shows `finest_level = 0` (coarse):
  - `tests/chk0000020/Header`
- Restart input uses `amr.n_cell = 64` and `restartRefineFactor_ = 2`:
  - `inputs/BinaryOrbit_refactor_splitparticle.in`
- `Particle_H` exists at coarse level:
  - `tests/chk0000020/CIC_particles/Level_0/Particle_H`
- Restart log shows dual-grid path followed by `locateParticle` abort.

## Proposed Solutions
### Option A (recommended)
Skip the dual-grid path on restart when `restartRefineFactor_ > 1` (or when the particle checkpoint grid does not match the refined grid). This forces a single-grid restart so particles are placed onto the refined grid directly.

**Implementation ideas:**
- Add a runtime knob (e.g., `particles.ignore_particle_header_on_restart = 1`) that bypasses `Particle_H` when refining.
- In `ParticleContainer_impl::Restart`, guard dual-grid behavior on grid compatibility or a passed-in flag.

### Option B
Regenerate or refine `Particle_H` to match the refined grid before calling `Restart`. This is more invasive and requires changes to restart workflows and metadata handling.

### Option C
Disable restart refinement in the failing test (workaround only).

## Risk Assessment
- Option A is localized to restart logic and avoids breaking normal non-refined restarts.
- Option B is higher risk and more complex.
- Option C does not address the root cause.

## Affected Tests
- `BinaryOrbitCICRefactor2`

## Status
Root cause identified. Fix not yet applied.
