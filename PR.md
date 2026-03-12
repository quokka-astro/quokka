## Add GravBonnorEbertSphere test problem

New test problem that initializes an exact Bonnor-Ebert sphere — an isothermal self-gravitating gas sphere in hydrostatic equilibrium — with typical star formation parameters (T=10K, μ=2.33 m_p).

### Physics

The Bonnor-Ebert sphere is computed by solving the isothermal Lane-Emden equation using RK4 integration. The critical sphere has dimensionless radius ξ_max ≈ 6.451 and density contrast ρ_c/ρ_edge ≈ 14.04.

### Test modes

- **Stability** (`overdensity_factor = 1.0`): The sphere remains approximately in hydrostatic equilibrium (~2% density change over 0.1 t_ff).
- **Collapse** (`overdensity_factor > 1.0`): The density is enhanced by the factor while pressure stays at equilibrium, so gravity overcomes pressure support and the central density increases.

### Files

- `src/problems/GravBonnorEbertSphere/testGravBonnorEbertSphere.cpp` — Problem implementation with Lane-Emden solver
- `src/problems/GravBonnorEbertSphere/CMakeLists.txt` — Build target (3D only)
- `inputs/GravBonnorEbertSphere.toml` — Input file with default parameters

### Runtime parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `problem.rho_c` | 3.0e-18 | Central density [g/cm³] |
| `problem.overdensity_factor` | 1.0 | Density enhancement (1.0=stable, >1=collapse) |
