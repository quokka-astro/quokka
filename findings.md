# Findings

## Key Architecture
- Star particle has 14 real components + 1 int component (burnState)
- `getParticleDataAtLevel()` returns `pair<vector<vector<double>>, vector<vector<int>>>`
- real_data[i] has positions first (3 elements) then all NReal rdata values
- So mass is at real_data[i][3], mdot at real_data[i][14], lum at real_data[i][16]
- StellarPhysics functions are marked AMREX_GPU_DEVICE but callable on host in CPU builds

## Critical: Timestep Call Order
1. timeStepWithSubcycling → accretion (computeSinkAccretion → applySinkAccretion) → updates mass/mdot
2. updateParticleProperties → reads mass/mdot → computes luminosity → stores lum
3. particleMeshInteraction → SN deposition

Wait, this is WRONG. Actual order is:
1. timeStepWithSubcycling (hydro advance, NO accretion)
2. updateParticleProperties → computes luminosity with PRE-ACCRETION mass/mdot
3. particleMeshInteraction → includes accretion → updates mass/mdot

So stored `lum` uses mass/mdot from BEFORE the last accretion step. Test uses 1% tolerance for this one-step lag.

## Root Cause: Tables Gate
`ParticlePropertyUpdateBase::updateParticleProperties` was gated on `g_luminosity_tables_ptr` being initialized. Star particles don't use tables, so they were never updated. Fixed by making StochasticStellarPop override `updateParticleProperties` to do the table check, while base runs unconditionally with empty tables.

## Burn State Transitions
Uninitialized → None → VariableCoreDeuterium → SteadyCoreDeuterium → ShellDeuterium → ZAMS
- Requires mass > M_rad_min (0.01 M_solar) AND mdot > 0 to leave Uninitialized
- Central temp > T_deuterium (1.5e6 K) to enter VariableCoreDeuterium
- In test: after 20 steps, burn_state=2 (VariableCoreDeuterium), mass~0.54 M_sun

## StellarPhysics Functions (host-callable in CPU builds)
- `rad_init(mdot)` — radius from accretion rate
- `n_init(mdot)` — polytropic index from accretion rate
- `luminosity_ZAMS(mass)` — Tout et al. (1996) ZAMS luminosity
- `luminosity_star(mass, radius, mdot)` — stellar luminosity with Hayashi limit
- `luminosity_disk(mass, radius, mdot)` — disk luminosity
- `luminosity_total(mass, radius, mdot, burn_state)` — total; 0 for Uninitialized
