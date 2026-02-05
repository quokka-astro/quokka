# Fix Galilean Invariance Bug in Sink Accretion

## Summary

Fixed a bug in the sink particle accretion module where the accretion rate was not Galilean invariant. The issue was that velocities used in the Bondi-Hoyle accretion formula were computed in the grid frame rather than the particle frame.

## Changes

### 1. particle_accretion.hpp

Updated `compute_Mdot_and_r_K` function to transform velocities to the particle frame:

The Bondi-Hoyle accretion rate depends on the relative velocity between the gas and the sink particle:

```
M_dot = 4π ρ_∞ r_BH² √(v_∞² + λ² c_s²)
```

where `r_BH = GM / (v_∞² + c_s²)` and `v_∞` is the gas velocity **relative to the particle**.

The bug computed `v_∞` as the absolute velocity in the grid frame, making the accretion rate frame-dependent. Particularly, the accretion rate was too low when the sink particle and ambient gas are moving at a high velocity. The fix transforms velocities to the particle frame, ensuring the accretion physics is Galilean invariant.

## Testing

The ParticleSink test validates the fix through three phases:

### Phase 1: Base simulation (1 timestep)
- Runs base case with zero boost velocity for a fixed timestep
- Validates density profile against analytical solution

### Phase 2: Boosted simulation (1 timestep) - Galilean invariance
- Runs boosted case with boost velocity of 1×10⁸ cm/s for a fixed timestep
- Compares density profile against analytical solution based on its actual evolution time
- Validates that Physics is Galilean invariant - both reference frames give equally accurate results

### Phase 3: Boosted simulation (10 more timesteps) - Mass conservation
- Continues boosted simulation for 10 additional timesteps
- Validates total mass conservation over multi-timestep evolution
- Validates that Mass conserved to machine precision
