# Findings

## Key Architecture
- Star particle has 14 real components + 1 int component (burnState)
- `getParticleDataAtLevel()` returns `pair<vector<vector<double>>, vector<vector<int>>>`
- real_data[i] has positions first (3 elements) then all NReal rdata values
- So mass is at real_data[i][3], mdot at real_data[i][14], lum at real_data[i][16]
- StellarPhysics functions are marked AMREX_GPU_DEVICE but callable on host in CPU builds

## Test Setup
- ParticleStar test: 32^3 box, reflecting BC, self-gravity, MHD
- Creates 1 Star particle from Jeans-unstable cell in step 1
- Runs 20 more steps of accretion
- Currently only validates mass conservation

## Burn State Transitions
Uninitialized → None → VariableCoreDeuterium → SteadyCoreDeuterium → ShellDeuterium → ZAMS
- Requires mass > M_rad_min (0.01 M_solar) AND mdot > 0 to leave Uninitialized
- Central temp > T_deuterium (1.5e6 K) to enter VariableCoreDeuterium
