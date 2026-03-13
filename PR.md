## Add dust to self-gravity + GravBonnorEbertSphere test

Adds dust density to the gravitational potential and validates with a new test problem.

### Core changes (`src/QuokkaSimulation.hpp`)

- **`fillPoissonRhsAtLevel`**: when `is_dust_enabled`, each dust group's density is added to the Poisson source term (4πGρ_dust), so dust mass contributes to the gravitational potential.
- **`applyPoissonGravityAtLevel`**: when `is_dust_enabled`, each dust group's momentum is kicked by `dt × ρ_dust × g` using the same gravitational acceleration as gas.

### New test problem (`src/problems/GravBonnorEbertSphere/`)

Initializes an exact Bonnor-Ebert sphere — an isothermal self-gravitating sphere in hydrostatic equilibrium — with:
- **Gas**: T = 10 K, μ = 2.33 m_p, ρ_c_gas = 2×10⁻¹⁸ g/cm³
- **Dust**: 2 groups with ρ_dust_total = ρ_gas (f = 1), tightly coupled (t_stop = 10⁸ s ≪ t_ff)

The equilibrium length scale accounts for the total (gas+dust) gravitational source:
```
r_0 = c_s / sqrt(4πG(1+f)ρ_c_total)
```
Gas and dust each provide 1/2 of total gravity.

### Test modes

| `overdensity_factor` | Expected | Result |
|---|---|---|
| 1.0 | Stable (~10% density drift) | −6.5% over 1 t_ff — PASS |
| 1.5 | Collapse (density increases >10%) | +414% — PASS |

### Runtime parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem.rho_c` | 3.0e-18 | Total central density [g/cm³] |
| `problem.overdensity_factor` | 1.0 | 1.0 = stable, >1 = collapse |
