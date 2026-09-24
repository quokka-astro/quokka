# SNFeedbackUtils::depositThermalKineticMomentumSNR(...): when `SN_smooth_gas_velocity == false`, the cross-term uses `((px*p_radial_x)+(py*p_radial_y)+(pz*p_radial_z))/rho` (`src/particles/particle_deposition.hpp:301-304`) without guarding `rho <= 0`

## Summary
when `SN_smooth_gas_velocity == false`, the cross-term uses `((px*p_radial_x)+(py*p_radial_y)+(pz*p_radial_z))/rho` (`src/particles/particle_deposition.hpp:301-304`) without guarding `rho <= 0`. In low-density/vacuum cells this can produce `inf`/`nan` energy deposition.

## Severity
`Medium`

## Affected File
`src/particles/particle_deposition.hpp`

## Affected Function / Symbol
`SNFeedbackUtils::depositThermalKineticMomentumSNR(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:503`
- Finding tags: robustness

## Proposed Patch
- Add explicit runtime guards for zero/non-finite/invalid inputs before the division or square-root path, and choose a deterministic fallback (early return, clamp, or hard abort) consistent with surrounding code.
