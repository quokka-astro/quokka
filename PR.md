# Add MHD-aware Jeans Density for Star and Sink Particle Formation

## Summary

This PR updates the particle recipe to use an MHD-aware Jeans density in star formation and sink particle formation. The effective sound speed is modified to account for magnetic pressure support against gravitational collapse.

## Changes

### Physics Update

The Jeans density formula is updated from:
```
rho_J = J^2 * pi * cs^2 / (G * dx^2)
```

to:
```
rho_J = J^2 * pi * cs^2 * (1 + 0.74/beta) / (G * dx^2)
```

where `beta` is the plasma beta (ratio of thermal pressure to magnetic pressure):
- `beta = P_thermal / P_magnetic`
- `P_thermal = rho * cs^2 / gamma` (from ideal gas law)
- `P_magnetic = B^2 / (8*pi)` in CGS units

The factor 0.74 accounts for the additional support provided by magnetic pressure in the Jeans instability criterion (see, e.g., Mouschovias & Spitzer 1976).

### Code Changes

1. **particle_utils.hpp**:
   - Updated `computeJeansDensity()` to accept an optional `plasma_beta` parameter (defaults to infinity for non-MHD cases)
   - Added `computePlasmaBeta()` helper function to compute plasma beta from thermal pressure and magnetic energy

2. **particle_creation.hpp**:
   - Updated Sink particle `ParticleChecker` to compute plasma beta and use MHD-aware Jeans density when MHD is enabled
   - Updated Sink particle `ParticleCreator` similarly

3. **particle_accretion.hpp**:
   - Updated `ComputeScaleDown()` to use MHD-aware Jeans density when limiting accretion rates

## Testing

- **ParticleSF test** (non-MHD case): Passed - verifies no regression for pure hydro simulations
- **ParticleSinkFormation test** (MHD case): Passed - verifies correct behavior with magnetic fields

The implementation uses `if constexpr` checks on `Physics_Traits<problem_t>::is_mhd_enabled` so there is no overhead for non-MHD simulations.
