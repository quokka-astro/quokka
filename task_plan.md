# Task Plan: GravBonnorEbertSphere + Dust Self-Gravity

## Goal
1. Add dust to the gravitational potential (Poisson RHS + dust gravity kick)
2. Validate with a Bonnor-Ebert sphere test that includes dust

## Status: COMPLETE

## Phases

### Phase 1: Research — COMPLETE
- [x] Study existing gravity/self-gravity test problems for patterns
- [x] Study fillPoissonRhsAtLevel and applyPoissonGravityAtLevel in QuokkaSimulation.hpp
- [x] Study DustDrag/dust_system pattern from DustDamping problem
- [x] Identify HydroSystem dust indices: dustDensity_index, x1DustMomentum_index, numDustVars_

### Phase 2: Core Implementation — COMPLETE
- [x] fillPoissonRhsAtLevel: add dust density to Poisson source (if is_dust_enabled)
- [x] applyPoissonGravityAtLevel: kick dust momentum by dt*rho_dust*g (if is_dust_enabled)
- [x] Committed: 2e41ebfb2

### Phase 3: BE Sphere Test with Dust — COMPLETE
- [x] Enable 2 dust groups (is_dust_enabled=true, nDustGroups=2)
- [x] DustDrag::ComputeReciprocalStoppingTime: short stopping time (tight coupling)
- [x] setInitialConditionsOnGrid: dust density = gas/2, split into 2 groups
- [x] Fix Lane-Emden length scale: use (1+f)*rho_c_total so total gravity balances pressure
- [x] Stability test: PASS (~6.5% drift over 1 t_ff)
- [x] Collapse test (1.5x overdensity): PASS (+414%)
- [x] Committed: e99232c49

### Phase 4: Documentation — COMPLETE
- [x] PR.md updated

## Decisions
- Lane-Emden length scale: r_0 = c_s/sqrt(4πG(1+f)ρ_c_total), NOT c_s/sqrt(4πGρ_c_total)
  - The (1+f) factor accounts for both gas and dust in the total gravitational source
  - Without this fix the sphere collapses immediately (gravity too strong for given pressure)
- Gas provides 2/3, dust provides 1/3 of total gravity with f=0.5 (not 1/2 each as user suggested)
  - Equal split would require f=1 (ρ_dust = ρ_gas)
- dust/DustDrag.hpp must NOT be included directly (it's pulled in by QuokkaSimulation.hpp)
- Valid BC names: "reflecting", "foextrap", "periodic" (NOT "outflow")
