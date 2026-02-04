# Fix momentum assignment for low-mass composite stars in stochastic star formation

## Summary

Fixed a bug in the star formation module where the total momentum of particles in a cell summed to zero instead of carrying the cell momentum.

## Problem

When high-mass stars are created (`num_particles > 1`), the low-mass composite star's velocity was set to cancel out the momentum of high-mass stars, resulting in zero total particle momentum. The particles should instead carry the cell momentum (total mass times cell velocity).

## Solution

Updated the momentum assignment for the low-mass composite star to ensure the center-of-mass (COM) velocity of all stars equals the cell velocity. The fix uses:

```cpp
plow.rdata(mass_idx + 1) = (real_particle_total_mass * vx - total_momx) / mass_low_mass_star;
```

This ensures correct velocities in a rotating disk, which is more important than strict momentum conservation given that mass conservation is already violated due to stochastic sampling of high-mass star masses from the IMF.

An alternative momentum-conserving option is included as commented code for reference.

## Files Changed

- `src/particles/particle_creation.hpp`: Fixed momentum assignment in `ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleCreator`
