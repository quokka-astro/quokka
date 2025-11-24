# Table of all test problems
This table lists all test problems in the Quokka codebase. The acronyms used are as follows:

- SG: Single-group radiation
- MG: Multi-group radiation
- ThermalDust: Dust thermally coupled to the gas
- PE: Photoelectric heating
- CIC: Cloud-in-cell particles

| Problem                  | DIM | Hydro | MHD | Rad               | Gravity | Particles                 | PassiveScalars |
|--------------------------|-----|-------|-----|-------------------|---------|---------------------------|----------------|
| Advection                | 1   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| Advection2D              | 2   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| AdvectionSemiellipse     | 1   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| AlfvenWaveCircular       | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| AlfvenWaveLinear         | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| BinaryOrbitCIC           | 3   | ✅     | ❌   | ❌                 | ✅       | CIC                       | ❌              |
| BrioWuShockTube          | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| Cooling                  | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| CurrentSheet             | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DiskGalaxy               | 3   | ✅     | ❌   | ❌                 | ✅       | CIC, StochasticStellarPop | ❌              |
| FCQuantities             | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| FastWave                 | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| FieldLoop                | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| GravRadParticle3D        | 3   | ❌     | ❌   | SG                | ✅       | CIC, Rad, CICRad          | ❌              |
| HydroBlast2D             | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroBlast3D             | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroContact             | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 2              |
| HydroHighMach            | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroKelvinHelmholz      | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroLeblanc             | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroQuirk               | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroRichtmeyerMeshkov   | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroSMS                 | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroShocktube           | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroShocktubeCMA        | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 3              |
| HydroShuOsher            | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroVacuum              | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroWave                | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroWaveConvergence     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| MHDBlast                 | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| MHDQuirk                 | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| NscbcChannel             | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| NscbcVortex              | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| ODEIntegration           | 1   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| OrszagTang               | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| ParticleAccretion        | 3   | ✅     | ❌   | ❌                 | ✅       | Sink                      | ❌              |
| ParticleCreation         | 3   | ✅     | ❌   | ❌                 | ✅       | Test                      | ❌              |
| ParticleRadiation        | 3   | ✅     | ❌   | MG                | ❌       | StochasticStellarPop      | ❌              |
| ParticleSF               | 3   | ✅     | ❌   | ❌                 | ❌       | StochasticStellarPop      | ❌              |
| ParticleSink             | 3   | ✅     | ❌   | ❌                 | ✅       | Sink                      | ❌              |
| ParticleSinkFormation    | 3   | ✅     | ❌   | ❌                 | ✅       | Sink                      | ❌              |
| PassiveScalar            | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| PopIII                   | 3   | ✅     | ❌   | ❌                 | ✅       | ❌                         | ❌              |
| PrimordialChem           | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| RadBeam                  | 2   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadDust                  | 1   | ✅     | ❌   | SG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadDustMG                | 1   | ✅     | ❌   | MG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadForce                 | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadLineCooling           | 1   | ✅     | ❌   | SG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadLineCoolingMG         | 1   | ✅     | ❌   | MG+ThermalDust+PE | ❌       | ❌                         | ❌              |
| RadMarshak               | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMarshakAsymptotic     | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMarshakCGS            | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMarshakDust           | 1   | ❌     | ❌   | MG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadMarshakDustPE         | 1   | ❌     | ❌   | MG+ThermalDust+PE | ❌       | ❌                         | ❌              |
| RadMarshakVaytet         | 1   | ❌     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadMatterCoupling        | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMatterCouplingRSLA    | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadShadow                | 2   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadStreaming             | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadStreamingY            | 2   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadSuOlson               | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadTophat                | 2   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadTube                  | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroBB               | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroPulse            | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroPulseDyn         | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroPulseGrey        | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroPulseMGconst     | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroPulseMGint       | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroShell            | 3   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroShock            | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroShockCGS         | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroShockMultigroup  | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroUniformAdvecting | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RandomBlast              | 3   | ✅     | ❌   | ❌                 | ✅       | ❌                         | 1              |
| RayleighTaylor2D         | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| RayleighTaylor3D         | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| ResampledCoolingTest     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| SN                       | 3   | ✅     | ❌   | ❌                 | ❌       | Test                      | ❌              |
| ShockCloud               | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 3              |
| SphericalCollapse        | 3   | ✅     | ❌   | ❌                 | ✅       | CIC                       | ❌              |
| StarCluster              | 3   | ✅     | ❌   | ❌                 | ✅       | ❌                         | ❌              |
