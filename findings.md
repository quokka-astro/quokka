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
- c_s = sqrt(k_B T / μ) ≈ 1.9e4 cm/s
- ρ_c ~ 10^-18 g/cm³
- r_0 = c_s / sqrt(4π G ρ_c)

## Codebase Patterns (from ParticleSink)
- Physics_Traits enables `is_self_gravity_enabled = true`
- EOS_Traits sets gamma and mean_molecular_weight
- `setInitialConditionsOnGrid` uses GPU ParallelFor
- ParmParse reads runtime parameters from input file
- CMakeLists: `quokka_add_problem(JOB_NAME Name)`
- Input file: TOML format with geometry, AMR, BC settings
- For self-gravity without particles: no Particle_Traits needed
- EOS: `ComputeEintFromTgas(rho, T)` for internal energy from temperature
- BCs: "reflecting" for isolated sphere, or "outflow"

## Implementation Strategy
- Solve Lane-Emden ODE on CPU at init, store in amrex::Gpu::DeviceVector
- Interpolate profile onto 3D grid in GPU kernel
- Uniform external medium at ρ_edge, T = 10K
- No particles, no MHD
- Overdensity factor as runtime parameter (1.0 = stable, >1 = collapse)
- Test: run a few timesteps, check density hasn't changed much (stable) or central density increased (collapse)
