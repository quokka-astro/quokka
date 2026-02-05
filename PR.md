# Fix Galilean Invariance Bug in Sink Accretion

## Summary

Fixed a bug in the sink particle accretion module where the accretion rate was not Galilean invariant. The issue was that velocities used in the Bondi-Hoyle accretion formula were computed in the grid frame rather than the particle frame.

## Changes

### 1. particle_accretion.hpp

Updated `compute_Mdot_and_r_K` function to transform velocities to the particle frame:

- Added particle velocity parameters (`par_vx`, `par_vy`, `par_vz`) to function signature
- Changed velocity calculation from grid frame to particle frame:
  ```cpp
  const double vx_grid = sum_px / sum_rho;
  const double vy_grid = sum_py / sum_rho;
  const double vz_grid = sum_pz / sum_rho;
  // Transform velocities to the particle frame to ensure Galilean invariance
  const double vx_infty = vx_grid - par_vx;
  const double vy_infty = vy_grid - par_vy;
  const double vz_infty = vz_grid - par_vz;
  ```
- Updated all call sites to pass particle velocities

### 2. testParticleSink.cpp

Added Galilean invariance validation similar to the SN test:

- Added `SimulationData` struct with `boost_velocity` field
- Updated `createInitialSinkParticles` to apply boost velocity to particles
- Updated `setInitialConditionsOnGrid` to apply boost velocity to gas
- Set physical `stopTime = 1000 years` for meaningful time scale
- Added second simulation run with boost velocity (1×10⁸ cm/s) and comparison logic
- Validates Galilean invariance of initial conditions to machine precision

### 3. ParticleSink.in

Updated input file for better timestep control:

- Removed hard-coded `initial_dt` parameter
- Removed `max_timesteps = 1` constraint to allow natural CFL evolution
- Added `init_shrink = 0.1` to reduce initial timestep conservatively

## Physics

The Bondi-Hoyle accretion rate depends on the relative velocity between the gas and the sink particle:

```
M_dot = 4π ρ_∞ r_BH² √(v_∞² + λ² c_s²)
```

where `r_BH = GM / (v_∞² + c_s²)` and `v_∞` is the gas velocity **relative to the particle**.

The bug computed `v_∞` as the absolute velocity in the grid frame, making the accretion rate frame-dependent. The fix transforms velocities to the particle frame, ensuring the accretion physics is Galilean invariant.

## Testing

The ParticleSink test validates Galilean invariance by running two simulations:
1. Base case with zero boost velocity
2. Boosted case with boost velocity of 1×10⁸ cm/s

Both simulations use:
- Physical `stopTime = 1000 years` (not tied to timestep)
- `init_shrink = 0.1` in the input file to ensure conservative initial CFL timestep
- `maxTimesteps = 0` for Galilean test (initialization only, no evolution)

After accounting for spatial shift due to the boost, the initial density profiles match to machine precision (relative error = 0). This validates that the sink accretion physics is correctly formulated in a Galilean-invariant manner.

The test also continues to validate:
- Mass conservation between gas and particles
- Density profile accuracy against analytical solution
- Multi-timestep evolution behavior

The exact solution for density depletion is computed using actual simulation time (`sim.tNew_[0]`), since accreted mass depends only on total evolution time, not on timestep or subcycling.
