# Add Passive Scalar Support to Supernova Feedback

## Summary

Adds support for depositing passive scalars (e.g., metals, dust) during supernova explosions. Scalars are deposited alongside mass, momentum, and energy using the same spatial kernel.

## Changes

### 1. Parameter Addition (`src/particles/particle_types.hpp`)
- Added global parameter `scalar_yield_per_SN` (default: 1.0)
- Extended `particleParmParse()` to read `particles.scalar_yield_per_SN` from input files
- Scalar yield represents total amount deposited per SN (not density)

### 2. Deposition Implementation (`src/particles/particle_deposition.hpp`)
- Modified `depositThermalSNR()` to deposit scalars when `numPassiveScalars > 0`
- Modified `depositThermalKineticMomentumSNR()` to deposit scalars consistently
- Scalars distributed using same kernel weights as mass/energy
- Used `constexpr if` for zero overhead when scalars disabled
- **Bug fix**: Corrected `count_comp` index to `HydroSystem<problem_t>::nvar_` instead of `Physics_NumVars::numHydroVars` to prevent overwriting scalar component
- Modified `addCompositeBufferToState()` and `addThermalOnlyBufferToState()` to transfer scalars from buffer to state
- **Note**: Currently only deposits to first passive scalar (`scalar0_index`); TODO added for future multi-scalar support

### 3. Test Validation (`src/problems/SN/testSN.cpp`)
- Enabled passive scalar (`numPassiveScalars = 1`)
- Initialize scalar field to `1e-6 * scalar_yield_per_SN / cell_vol`
- Validate total scalar conservation (< 1e-10 relative error)
- Validate peak scalar enhancement (> 10× initial density)
- Return non-zero status if validation fails

## Verification

The implementation correctly:
1. Deposits scalars with the same spatial distribution as mass
2. Conserves total scalar to machine precision
3. Creates localized scalar enhancements at SN sites
4. Works with all SN schemes (thermal-only, thermal+momentum, thermal+kinetic)
5. Has zero overhead when passive scalars disabled via `constexpr if`
