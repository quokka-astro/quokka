# Known Issues and Errata

This page collects current limitations, beta-status physics features, and any corrections to the published documentation or methods papers.

## Beta physics modules

The following Quokka physics modules should currently be treated as **beta** because they have not yet been exercised in a published science application with Quokka:

| Module | Status | Notes | Documentation |
| ------ | :----: | ----- | ------------- |
| Radiation | beta | Two-moment radiation transport and matter-radiation coupling | [Equations](equations.md), [Radiation Integrator](radiation_integrator.md) |
| Magnetohydrodynamics (MHD) | beta | Ideal MHD with constrained transport | [MHD module](mhd_module.md) |
| Dust | beta | Dedicated dust dynamics and dust-gas drag source terms | [Dust module](dust_module.md) |
| Particles | beta | Particle-mesh gravity, sink particles, star formation, and feedback | [Particles](particles.md) |
| Chemistry | beta | Primordial chemistry source terms | [Equations](equations.md), [Runtime parameters](parameters.md) |
| Self-gravity | beta | Poisson solve for gas and particle mass | [Equations](equations.md) |

Hydrodynamics and optically-thin cooling are not currently marked as beta.

## Current limitations

- `MHD + radiation` is not yet tested and is currently explicitly disabled in the code.
- Reflecting magnetic-field boundary conditions are not yet physically complete; Quokka currently applies `reflect_even` to all magnetic-field components.
- `MHD + dust` is not exercised by an in-tree problem and should be treated as untested.
- Particle features require a 3D build (`-DAMReX_SPACEDIM=3`).

## Errata

No published errata are currently recorded on this page.

If you find a documentation error or a discrepancy between the documented equations and the implementation, please open an issue on the Quokka GitHub repository.
