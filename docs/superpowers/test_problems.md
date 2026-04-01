## Problems to test against

| Problem                  | DIM | Hydro | MHD | Rad               | Gravity | Particles            | Comments                                                              |
|--------------------------|-----|-------|-----|-------------------|---------|----------------------|-----------------------------------------------------------------------|
| RadStreaming             | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | Simplest radiation transport test                                     |
| RadhydroShockCGS         | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | Most stringent test of the RHD solver                                 |
| RadhydroShockMultigroup  | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | Multi-group version of RadhydroShockCGS                               |
| RadMarshakVaytet         | 1   | ❌    | ❌  | MG                | ❌      | ❌                   |                                                                       |
| RadMarshakAsymptotic     | 1   | ❌    | ❌  | SG                | ❌      | ❌                   |                                                                       |
| RadhydroUniformAdvecting | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | Test of diffusion accuracy                                            |
| RadhydroPulseMGint       | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | Test of MG RHD with frequency-dependent opacity at each band          |
| RadhydroPulseMGconst     | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | Test of MG RHD with constant opacity at each band                     |
| RadMarshakDust           | 1   | ❌    | ❌  | MG+ThermalDust    | ❌      | ❌                   |                                                                       |
| RadMarshakDustPE         | 1   | ❌    | ❌  | MG+ThermalDust+PE | ❌      | ❌                   |                                                                       |
| RadDust                  | 1   | ✅    | ❌  | SG+ThermalDust    | ❌      | ❌                   |                                                                       |
| RadDustMG                | 1   | ✅    | ❌  | MG+ThermalDust    | ❌      | ❌                   |                                                                       |
| RadForce                 | 1   | ✅    | ❌  | SG                | ❌      | ❌                   |                                                                       |
| RadTube                  | 1   | ✅    | ❌  | MG                | ❌      | ❌                   |                                                                       |
| RadhydroBB               | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | Test the accuracy of piecewise powerlaw model                         |
| ParticleRadiation        | 3   | ✅    | ❌  | MG                | ❌      | StochasticStellarPop | 3D test with radiating particles; test the external source term `Src` |

## The complete list of problems with radiation turned on, for reference 

Please ignore.

| Problem                  | DIM | Hydro | MHD | Rad               | Gravity | Particles            | PassiveScalars |
|--------------------------|-----|-------|-----|-------------------|---------|----------------------|----------------|
| GravRadParticle3D        | 3   | ❌    | ❌  | SG                | ✅      | CIC, Rad, CICRad     | ❌             |
| ParticleRadiation        | 3   | ✅    | ❌  | MG                | ❌      | StochasticStellarPop | ❌             |
| RadBeam                  | 2   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadDust                  | 1   | ✅    | ❌  | SG+ThermalDust    | ❌      | ❌                   | ❌             |
| RadDustMG                | 1   | ✅    | ❌  | MG+ThermalDust    | ❌      | ❌                   | ❌             |
| RadForce                 | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadLineCooling           | 1   | ✅    | ❌  | SG+ThermalDust    | ❌      | ❌                   | ❌             |
| RadLineCoolingMG         | 1   | ✅    | ❌  | MG+ThermalDust+PE | ❌      | ❌                   | ❌             |
| RadMarshak               | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadMarshakAsymptotic     | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadMarshakCGS            | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadMarshakDust           | 1   | ❌    | ❌  | MG+ThermalDust    | ❌      | ❌                   | ❌             |
| RadMarshakDustPE         | 1   | ❌    | ❌  | MG+ThermalDust+PE | ❌      | ❌                   | ❌             |
| RadMarshakVaytet         | 1   | ❌    | ❌  | MG                | ❌      | ❌                   | ❌             |
| RadMatterCoupling        | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadMatterCouplingRSLA    | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadShadow                | 2   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadStreaming             | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadStreamingY            | 2   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadSuOlson               | 1   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadTophat                | 2   | ❌    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadTube                  | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | ❌             |
| RadhydroBB               | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | ❌             |
| RadhydroPulse            | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadhydroPulseDyn         | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadhydroPulseGrey        | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadhydroPulseMGconst     | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | ❌             |
| RadhydroPulseMGint       | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | ❌             |
| RadhydroShell            | 3   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadhydroShock            | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadhydroShockCGS         | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
| RadhydroShockMultigroup  | 1   | ✅    | ❌  | MG                | ❌      | ❌                   | ❌             |
| RadhydroUniformAdvecting | 1   | ✅    | ❌  | SG                | ❌      | ❌                   | ❌             |
