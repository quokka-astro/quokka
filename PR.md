## Summary

Refactor `updateParticleProperties` in `PhysicsParticles.hpp` to use the traits class pattern, consistent with how `createParticlesFromState` delegates to `ParticleCreationTraits`.

- **`particle_update.hpp`**: Added a container-level `updateParticleProperties` static method to both the default `ParticlePropertyUpdateTraits` (no-op) and the `StochasticStellarPop` specialization (moves the GPU table lookup and particle iteration loop from `PhysicsParticles.hpp` into the traits class). Added `AMReX_BLProfiler.H` and `physics_info.hpp` includes.
- **`PhysicsParticles.hpp`**: Replaced the 30-line inline implementation of `updateParticleProperties` with a single-line delegation to `ParticlePropertyUpdateTraits<particleType>::template updateParticleProperties<problem_t, ContainerType>`.

This makes `PhysicsParticles.hpp` agnostic to particle-type-specific update logic, preparing for future implementation of star particles.

## Test plan

- [x] Build `ParticleRadiation` test — no errors
- [x] Run `ParticleRadiation` test — passes with relative error `< 1e-15`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
