# Findings: GravBonnorEbertSphere

## Bonnor-Ebert Sphere Physics
- Isothermal self-gravitating gas sphere in hydrostatic equilibrium
- Described by isothermal Lane-Emden equation: (1/ξ²) d/dξ(ξ² dψ/dξ) = e^(-ψ)
- ψ = ln(ρ_c/ρ), ξ = r/r_0 where r_0 = c_s/√(4πGρ_c)
- Critical dimensionless radius: ξ_max ≈ 6.451
- Critical density contrast: ρ_c/ρ_edge ≈ 14.04
- Stable if ξ_max < 6.451, unstable (collapses) if overdense

## Typical Star Formation Parameters
- T = 10 K, μ = 2.33 m_p (molecular H₂ + He)
- c_s = sqrt(k_B T / μ) ≈ 1.88e4 cm/s
- ρ_c = 3e-18 g/cm³ → r_0 ≈ 1.19e16 cm (0.00385 pc)
- R_sphere = 6.451 * r_0 ≈ 7.65e16 cm (0.0248 pc)
- t_ff ≈ 1.21e12 s (38,432 yr), t_sc ≈ 4.07e12 s (128,871 yr)

## Codebase Patterns (from ParticleSink / SphericalCollapse)
- Physics_Traits enables `is_self_gravity_enabled = true`
- EOS_Traits sets gamma and mean_molecular_weight
- `setInitialConditionsOnGrid` uses GPU ParallelFor
- ParmParse reads runtime parameters from input file
- CMakeLists: `quokka_add_problem(JOB_NAME Name)`
- Input file: TOML format with geometry, AMR, BC settings
- Valid BC names: `periodic`, `reflecting`, `foextrap`, `ext_dir` (NOT `outflow`)
- `C::parsec` for parsec constant (not `C::pc`)
- EOS: `ComputeEintFromPres(rho, P)` for internal energy from pressure

## Key Implementation Details
- Lane-Emden solved on CPU with RK4 (10000 points), copied to GPU DeviceVector
- Profile interpolated with linear interpolation in GPU kernel
- Overdensity factor multiplies density but NOT pressure → breaks equilibrium
- External medium at edge density and pressure (pressure_contrast = 1)
- gamma = 1.001 approximates isothermal behavior
