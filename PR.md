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

Added explicit ghost cell filling after both MLMG and OpenBC Poisson solves in `calculateGpotAllLevels()`:

### Base Level (lev=0):
1. **`FillBoundary()`** - Fills ghost cells at periodic boundaries and fine-fine interfaces
2. **`setFunctorPhiZero` + `PhysBCFunct`** - For MLMG solver ONLY, sets ghost cells to φ = 0 at physical boundaries (homogeneous Dirichlet BC)

### Fine Levels (lev>0):
1. **`FillBoundary()`** - Fills periodic boundaries and fine-fine interfaces  
2. **Physical BCs** - For MLMG solver only, applies Dirichlet BC (φ = 0)
3. **NOTE**: Coarse-fine boundary ghost cells are not explicitly interpolated from coarse level in this implementation. The MLMG solver ensures consistency in valid cells, but ghost cells at coarse-fine boundaries rely on extrapolation from `FillBoundary()`. For better accuracy, `FillPatchTwoLevels` should be used (marked as TODO).

**Key distinctions:**
- **MLMG with Dirichlet BC**: Ghost cells at physical boundaries are set to φ = 0
- **OpenBC solver**: Ghost cells at physical boundaries retain solver-computed values (φ → 0 at infinity, NOT at domain boundaries)

Also updated `kickParticlesAllLevels()` to use the same boundary handling approach for consistency.

## Code Changes

- **New functor**: `setFunctorPhiZero` - Enforces homogeneous Dirichlet BC (φ = 0) at physical boundaries for MLMG solver
- **Ghost cell filling**: Added after `mlmg.solve()` and `poissonSolver.solve()` in `calculateGpotAllLevels()`
- **OpenBC handling**: Physical boundary ghost cells are NOT set to zero for OpenBC solver (correct behavior)
- **AMR limitation**: Ghost cells at coarse-fine boundaries use `FillBoundary()` extrapolation (TODO: implement `FillPatchTwoLevels` for proper interpolation)
