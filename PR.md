# Add runtime option for Galilean-invariant vs energy-conserving SN feedback

## Summary

This PR adds a new runtime parameter `particles.SN_use_galilean_invariant` that allows users to choose between:

1. **Galilean-invariant SN feedback** (default, `particles.SN_use_galilean_invariant = 1`):
   - Uses center-of-mass (COM) frame formulation
   - Computes COM velocity of SNR (gas + ejecta)
   - Deposits momentum such that cells get velocity = v_COM + v_radial
   - Includes cross term (v_COM · p_radial) in energy deposition for Galilean invariance
   - This formulation ensures that the SN feedback is invariant under Galilean transformations

2. **Energy-conserving SN feedback** (legacy, `particles.SN_use_galilean_invariant = 0`):
   - Uses lab-frame formulation
   - Momentum change preserves cell velocity proportionally and adds radial momentum
   - Energy deposition uses lab-frame ejecta kinetic energy
   - This is the original implementation from the common base

## Changes

### Modified files:
- `src/particles/particle_types.hpp`:
  - Added global parameter `SN_use_galilean_invariant` (default: `true`)
  - Added parsing of this parameter in `particleParmParse()`

- `src/particles/particle_deposition.hpp`:
  - Modified `depositThermalKineticMomentumSNR()` to accept `use_galilean_invariant` parameter
  - Added conditional logic to choose between Galilean-invariant and energy-conserving formulations
  - Updated call site in `depositToBuffer()` to pass the global parameter

- `inputs/SN.in`:
  - Added `particles.SN_use_galilean_invariant = 1` parameter

- `inputs/RandomBlast.in`:
  - Added `particles.SN_scheme` and `particles.SN_use_galilean_invariant` parameters

## Testing

The SN test passes with both settings:
- Galilean-invariant (default): Test passes with expected relative errors
- Energy-conserving: Test passes with different (lower in this case) relative errors

## Notes

- The parameter only affects momentum-based SN schemes (`SN_thermal_or_thermal_momentum`, `SN_thermal_kinetic_or_thermal_momentum`, `SN_pure_kinetic_or_thermal_momentum`)
- The `SN_thermal_only` scheme always uses energy-conserving formulation (no momentum deposition)
- Default setting maintains the new Galilean-invariant behavior implemented in this branch
- Setting `particles.SN_use_galilean_invariant = 0` reverts to the original energy-conserving behavior from the development branch
