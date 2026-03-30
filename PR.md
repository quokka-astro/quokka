Adds a Gaussian kernel particle-to-mesh interpolator with spherical radial cutoff for radiation deposition, replacing the linear (CIC) interpolator.

## Changes

### New: `ParticleInterpolator::Gaussian<N>` (`src/particles/particle_deposition.hpp`)

A standalone interpolator (does not inherit from AMReX `Base` since the spherical cutoff breaks separability):

- Template parameter `N` (default 3) controls cutoff radius: kernel weight is non-zero only within a sphere of radius `N·dx` centered on the particle
- `static constexpr amrex::Real sigma = 1.5` (in units of dx) controls the Gaussian width
- Weights are normalized over the spherical region (sum = 1), ensuring strict energy conservation
- `exp(-r²/(2σ²))` computed on-the-fly to avoid storing `(2N+1)³` weights on GPU stack

### Updated: `RadDeposition` (`src/particles/particle_deposition.hpp`)

Switched from `ParticleInterpolator::Linear` to `ParticleInterpolator::Gaussian<>` (N=3).

### Updated: `QuokkaSimulation` (`src/QuokkaSimulation.hpp`)

Increased `radEnergySource` ghost cell count from 1 to 4 (`N + 1`) to accommodate the wider stencil. The `+1` accounts for possible particle drift outside the valid box before deposition.

## Validation

All three `ParticleRadiation` test variants pass with relative energy conservation errors of ~10⁻¹⁶, well within the 10⁻¹³ tolerance.

## Checklist
_Before this pull request can be reviewed, all of these tasks should be completed. Denote completed tasks with an `x` inside the square brackets `[ ]` in the Markdown source below:_
- [x] I have added a description (see above).
- [x] I have added a link to any related issues (if applicable; see above).
- [x] I have read the [Contributing Guide](https://github.com/quokka-astro/quokka/blob/development/CONTRIBUTING.md).
- [ ] I have added tests for any new physics that this PR adds to the code.
- [ ] *(For quokka-astro org members)* I have manually triggered the GPU tests with the magic comment `/azp run`.
