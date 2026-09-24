# RadSystem::ComputeCellOpticalDepth<DIR>(...): harmonic-mean optical depth uses `2*tau_L*tau_R/(tau_L+tau_R)` with no zero-denominator guard (`src/radiation/radiation_system.hpp:914-923`)

## Summary
harmonic-mean optical depth uses `2*tau_L*tau_R/(tau_L+tau_R)` with no zero-denominator guard (`src/radiation/radiation_system.hpp:914-923`). If both sides are optically thin with zero opacity/depth, this becomes `0/0` and can inject NaNs into the optional wavespeed-correction path.

## Severity
`Medium`

## Affected File
`src/radiation/radiation_system.hpp`

## Affected Function / Symbol
`RadSystem::ComputeCellOpticalDepth<DIR>(...)`

## Audit Metadata
- Source log: `audit/src-audit-log.md:408`
- Finding tags: robustness

## Proposed Patch
- Apply a targeted fix at the cited location, then add a regression/unit test that exercises the failing code path and confirms the corrected behavior.
