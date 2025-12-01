# Self-Gravity Implementation in Quokka

This directory contains the implementation of self-gravity for Quokka using Approach 1 from the [Castro/Nyx gravity paper](https://arxiv.org/pdf/1005.0114.pdf), which is the same approach used by Enzo.

## Overview

The implementation provides self-gravity support through the Poisson equation:
```
∇²φ = 4πGρ
```

where φ is the gravitational potential, G is the gravitational constant, and ρ is the mass density.

## Approach 1: Enzo Method

This implementation uses **Approach 1** from the literature, which performs:
- **1 Poisson solve per level advance** (simplest method)
- **No sync solves** or composite solves
- **Level-by-level solving** with boundary conditions interpolated from coarser levels

### Algorithm Summary
1. After hydro update on level `l`, solve Poisson equation on level `l` only
2. Compute gravitational acceleration: **g** = -∇φ  
3. Apply gravitational forces in operator-split fashion:
   - Update momentum: **Δ(ρu)** = ρ**g**Δt
   - Update energy: **ΔE** = (**p**_old + 0.5**Δp**) · **g**Δt

## Files

### Core Implementation
- `PoissonGravity.hpp` - Main class interface and declarations
- `PoissonGravity_impl.hpp` - Template implementation of all methods

### Integration
- Modified `QuokkaSimulation.hpp` - Added PoissonGravity member and includes
- Modified `QuokkaSimulation.cpp` - Implemented `fillPoissonRhsAtLevel()` and `applyPoissonGravityAtLevel()`

### Test Problem
- `../problems/SelfGravityTest/` - Simple self-gravitating gas cloud test

## Key Features

### Level-by-Level Poisson Solving
- Each AMR level solves its own Poisson equation independently
- Uses AMReX's `MLPoisson` solver with single-level grids
- Supports periodic and Dirichlet boundary conditions

### Boundary Condition Handling
- **Level 0**: Free-space boundary conditions (Dirichlet φ=0 at domain boundaries)
- **Level > 0**: Dirichlet boundary conditions interpolated from level-1 solution
- Automatic detection of periodic domains

### Operator Splitting
- Gravitational forces applied after hydrodynamics update
- Momentum update: explicit force application
- Energy update: accounts for work done by gravitational forces

## Compilation

Enable gravity support during CMake configuration:
```bash
cmake .. -DQUOKKA_GRAVITY=ON -DAMReX_SPACEDIM=3
```

This adds the `QUOKKA_USE_GRAVITY` compile-time definition.

## Usage in Problems

Enable self-gravity in your problem by setting the physics trait:
```cpp
template <> struct Physics_Traits<YourProblem> {
    static constexpr bool is_self_gravity_enabled = true;
    static constexpr double gravitational_constant = 1.0; // or C::Gconst for CGS
    // ... other traits
};
```

## Input Parameters

Configure gravity solver in input files:
```
gravity.gravitational_constant = 6.67430e-8  # CGS units
gravity.tolerance = 1.0e-12                  # solver tolerance  
gravity.max_iterations = 200                 # max solver iterations
gravity.verbose = 1                          # verbosity level
```

## Limitations

Current implementation limitations (compared to Approaches 2 and 3):
- **No sync solves**: May cause inconsistencies at coarse-fine boundaries during advection
- **No composite solves**: Less accurate than multilevel solves
- **Simple boundary conditions**: Free-space BCs approximated by Dirichlet φ=0

These limitations make this approach simpler but potentially less accurate for problems where structures advect across AMR boundaries.

## Future Enhancements

Potential improvements:
1. **Approach 2**: Add sync solves after reflux operations
2. **Approach 3**: Add composite multilevel solves with correction terms
3. **Better boundary conditions**: Implement true free-space boundary conditions
4. **Time interpolation**: Handle time-dependent boundary condition interpolation

## References

- Castro hydro+gravity paper: https://arxiv.org/pdf/1005.0114.pdf
- Miniati & Colella 2007: ORION2 gravity implementation
- AMReX documentation: Linear solvers guide