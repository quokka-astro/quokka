# Task Plan: Validate Star Particle updateProperties & Fix Bugs

## Goal
1. Fix bugs in `starparticle_radiation.hpp` (luminosity never stored, missing initializations)
2. Refactor stellar physics functions for reusability in test code
3. Update `testParticleStar.cpp` to validate stellar property updates (burning state, luminosity, polytropic index)

## Findings

### Bugs Identified
1. **Luminosity never stored**: `updateStellarPropertiesImpl()` computes luminosity for state transitions but never writes it to `p.rdata(StarParticleLumIdx)`. The `lum` field (index 13) is never set after creation.
2. **birth_time not initialized**: Star particle creation (particle_creation.hpp:757-768) doesn't set `p.rdata(StarParticleBirthTimeIdx)` (index 4) or `p.rdata(StarParticleDeathTimeIdx)` (index 5). These are left uninitialized.
3. **lum not initialized**: Star particle creation doesn't initialize `p.rdata(StarParticleLumIdx)` (index 13) to 0.
4. **Duplicate BurningState enum**: `particle_types.hpp` has unscoped `burningState`, `starparticle_radiation.hpp` has scoped `StellarPhysics::BurningState`. The impl uses the unscoped one; `luminosity_total` uses the scoped one.
5. **Early return bug**: When `burn_state == Uninitialized` and `mass < M_rad_min || mdot == 0.0`, the function returns without writing back `mdeut` (which was modified on line 446).

### Data Layout (getParticleDataAtLevel)
- `real_data[i]` = [pos_x, pos_y, pos_z, mass, vx, vy, vz, birth_time, death_time, amx, amy, amz, mdeut, n, mdot, l_hist, lum]
- `int_data[i]` = [burnState]
- Offsets: position adds 3, so `real_data[i][3 + StarParticleXxxIdx]`

## Phases

### Phase 1: Fix bugs in starparticle_radiation.hpp `[pending]`
- [ ] Fix early return: write mdeut back before returning, or don't modify mdeut before the check
- [ ] Add luminosity computation and storage at end of `updateStellarPropertiesImpl`
- [ ] Remove duplicate `StellarPhysics::BurningState` enum, use the one from `particle_types.hpp`

### Phase 2: Fix particle creation initialization `[pending]`
- [ ] Initialize `birth_time`, `death_time`, `lum` in Star particle creation (particle_creation.hpp)

### Phase 3: Update test to validate stellar properties `[pending]`
- [ ] Include `starparticle_radiation.hpp` in test file
- [ ] After evolution, extract particle data via `getParticleDataAtLevel()`
- [ ] Validate burning state, polytropic index n, luminosity against host-side computation using StellarPhysics functions
- [ ] Print diagnostics for stellar properties

### Phase 4: Build and run test `[pending]`

- [ ] Build ParticleStar test
- [ ] Run and verify all checks pass

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
