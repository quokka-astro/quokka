# SN Passive Scalar Deposition Design

**Date**: 2026-02-20  
**Status**: Approved

## Overview

Add support for passive scalar deposition in supernova feedback, allowing SNe to inject tracer quantities (metals, dust, etc.) into the ISM alongside mass, momentum, and energy.

## Motivation

Current SN feedback deposits mass, momentum, and energy but lacks support for passive scalars (e.g., metal enrichment, dust injection). This feature enables tracking chemical evolution and other tracer quantities through SN enrichment.

## Design

### 1. Parameter Addition

**File**: `src/particles/particle_types.hpp`

Add global parameter:
```cpp
inline amrex::Real scalar_yield_per_SN = 1.0; // NOLINT
```

Extend `particleParmParse()`:
```cpp
pp.query("scalar_yield_per_SN", scalar_yield_per_SN);
```

**Input parameter**: `particles.scalar_yield_per_SN` (default: 1.0)
- Units: Total scalar amount per SN (not density)
- Divided by cell volume during deposition to get scalar density

### 2. Deposition Implementation

**File**: `src/particles/particle_deposition.hpp`

Modify both `depositThermalSNR()` and `depositThermalKineticMomentumSNR()`:

```cpp
// Inside stencil loop, after energy deposition:
if constexpr (Physics_Traits<problem_t>::numPassiveScalars > 0) {
    const amrex::Real scalar_per_cell = scalar_yield_per_SN * kernel_times_vol_inverse;
    amrex::Gpu::Atomic::AddNoRet(&local_buffer(ix + ii, iy + jj, iz + kk, 
                                  HydroSystem<problem_t>::scalar0_index), 
                                  scalar_per_cell);
}
```

**Key points**:
- Use same spatial kernel as mass/energy for consistency
- `constexpr if` ensures zero overhead when scalars disabled
- Deposits to first passive scalar (`scalar0_index`)

### 3. Test Validation

**File**: `src/problems/TallBoxSf/testTallBoxSf.cpp`

**Initial conditions**:
```cpp
state_cc(i, j, k, HydroSystem<TheProblem>::scalar0_index) = 
    1e-6 * scalar_yield_per_SN / cell_vol;
```

**Validation in `problem_main()`**:
1. Compute initial total scalar: `sum(scalar_density * cell_vol)`
2. After evolution, compute final total scalar
3. Compute peak scalar density: `max(scalar_density)`
4. Check conservation: `|final - initial - 2*scalar_yield_per_SN| / (initial + 2*scalar_yield) < 1e-10`
5. Check peak enhancement: `peak_scalar > 10 * initial_scalar_density`

**Test setup**: TallBoxSf with 2 SNe provides validation scenario

## Implementation Details

### Spatial Distribution
- Scalars use identical kernel weights as mass deposition
- Ensures physical consistency (scalar follows mass)
- Kernel: top-hat smoothed over 3-cell radius stencil

### Numerical Considerations
- Atomic operations prevent race conditions in parallel deposition
- Buffer accumulation + boundary sum ensures conservation across MPI ranks
- Roundoff algorithm maintains numerical precision

### Compatibility
- Feature is compile-time optional via `constexpr if`
- Zero overhead when `numPassiveScalars == 0`
- Works with all SN schemes (thermal-only, thermal+momentum, thermal+kinetic)

## Testing

**Primary test**: TallBoxSf
- 2 SN explosions
- Initial scalar: `1e-6 * scalar_yield_per_SN / cell_vol`
- Validates conservation to machine precision (1e-10)
- Validates peak enhancement (>10× initial)

## Success Criteria

1. Code compiles with and without passive scalars enabled
2. TallBoxSf test passes conservation check (< 1e-10 relative error)
3. Peak scalar density exceeds 10× initial value
4. No performance regression when scalars disabled
5. Works with all SN schemes (thermal-only, momentum, kinetic)
