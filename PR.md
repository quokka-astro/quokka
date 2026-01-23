# Fix Ghost Cell Handling for Gravitational Potential After Poisson Solve

## Problem

The MLMG Poisson solver (`mlmg.solve()`) only fills valid (interior) cells of the solution MultiFab `phi`, leaving ghost cells uninitialized. This caused incorrect gravitational accelerations at box boundaries when computing gradients in:
- `applyPoissonGravityAtLevel()` - applies gravity to hydrodynamic state
- `kickParticlesAllLevels()` - computes particle acceleration from potential gradients

Both functions use centered finite differences that require properly filled ghost cells:
```cpp
gx = -0.5 * (phi[i+1] - phi[i-1]) / dx
```

## Solution

### `calculateGpotAllLevels()` - Ghost Cell Filling After Poisson Solve

**For MLMG solver** (any dimension is periodic):
- Fill periodic boundaries with `FillBoundary()`
- Apply Dirichlet BC (φ = 0) at non-periodic physical boundaries using `setFunctorPhiZero`
- Coarse-fine boundaries are handled by the MLMG solver for valid cells

**For OpenBC solver** (no periodic dimensions):
- No ghost cell filling needed - the solver computes physically-consistent boundary values (φ → 0 at infinity, NOT at domain boundaries)

### `kickParticlesAllLevels()` - Extended Ghost Cells for Particle Acceleration

Particles require `phi_extended` with 3 ghost cells (vs 1 in `phi`) for gradient computation with CIC interpolation.

**Level 0:**
- Copy from `phi`, fill periodic boundaries with `FillBoundary()`
- For MLMG with Dirichlet BC: apply `setFunctorPhiZero` at non-periodic boundaries

**Fine levels:**
- Use `FillPatchTwoLevels` to properly handle coarse-fine interpolation
- For MLMG with Dirichlet BC: use `setFunctorPhiZero` boundary functor
- For fully periodic or OpenBC: use no-op functor `setFunctorParticleAccel`

## Code Changes

- **New functor `setFunctorPhiZero`**: Enforces homogeneous Dirichlet BC (φ = 0) at physical boundaries
- **New functor `setFunctorParticleAccel`**: No-op functor for OpenBC/periodic cases in particle acceleration
- **`calculateGpotAllLevels()`**: Ghost cell filling moved inside `if (use_mlmg_solver)` block; OpenBC needs no filling
- **`kickParticlesAllLevels()`**: Uses `FillPatchTwoLevels` for fine levels with proper coarse-fine interpolation
