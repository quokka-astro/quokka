# Table of all test problems
This table lists all test problems in the Quokka codebase. The acronyms used are as follows:

- SG: Single-group radiation
- MG: Multi-group radiation
- ThermalDust: Dust thermally coupled to the gas
- PE: Photoelectric heating
- CIC: Cloud-in-cell particles

| Problem                           | DIM | Hydro | MHD | Rad               | Gravity | Particles                 | PassiveScalars |
|-----------------------------------|-----|-------|-----|-------------------|---------|---------------------------|----------------|
| Advection                         | 1   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| AdvectionSemiellipse              | 1   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| AlfvenWaveCircularConvergence     | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| AlfvenWaveLinearConvergence       | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| BinaryOrbitCIC                    | 3   | ✅     | ❌   | ❌                 | ✅       | CIC                       | ❌              |
| BrioWuShockTube                   | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| CurrentSheet                      | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DTypeFront                        | 3   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| DTypeFrontVC                      | 2   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| DiskGalaxy                        | 3   | ✅     | ✅   | ❌                 | ✅       | CIC, StochasticStellarPop | 1              |
| DustAdvection                     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| DustAdvection3D                   | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| DustDampedGyromotion              | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustDamping                       | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| DustDampingIteration              | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| DustDampingIterationMHDZeroCharge | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustDampingMHDZeroB               | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustDampingWithExternalForce      | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| DustHallPedersenDrift             | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustLorentzShock                  | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustMagnetizedRDI                 | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustSoundwave                     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| DustyAlfvenWave                   | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustyOrszagTang                   | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| DustyShock                        | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| EntropyWaveConvergence            | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| FCQuantities                      | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| FastWaveConvergence               | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| FieldLoop                         | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| GravRadParticle3D                 | 3   | ❌     | ❌   | SG                | ✅       | CIC, Rad, CICRad          | ❌              |
| HydroBlast3D                      | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroContact                      | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 2              |
| HydroHighMach                     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroLeblanc                      | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroQuirk                        | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroSMS                          | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroShocktube                    | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroShocktubeCMA                 | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 3              |
| HydroShuOsher                     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroVacuum                       | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydroWaveConvergence              | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| HydrostaticAtmosphere             | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| MHDBalsaraVortex                  | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| MHDBitwiseICs                     | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| MHDBlast                          | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| MHDQuirk                          | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| MHDResistiveEnergyFluxKernel      | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| NscbcChannel                      | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| NscbcVortex                       | 2   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| ODEIntegration                    | 1   | ❌     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| OneZonePhotoionization            | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| OrszagTang                        | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| ParticleAccretion                 | 3   | ✅     | ✅   | ❌                 | ✅       | Sink                      | ❌              |
| ParticleCreation                  | 3   | ✅     | ✅   | ❌                 | ✅       | Test                      | ❌              |
| ParticleDeposition                | 3   | ✅     | ❌   | ❌                 | ❌       | CIC, Test                 | ❌              |
| ParticleRadiation                 | 3   | ✅     | ❌   | MG                | ❌       | StochasticStellarPop      | ❌              |
| ParticleSF                        | 3   | ✅     | ❌   | ❌                 | ❌       | StochasticStellarPop      | ❌              |
| ParticleSink                      | 3   | ✅     | ✅   | ❌                 | ✅       | Sink                      | ❌              |
| ParticleSinkFormation             | 3   | ✅     | ✅   | ❌                 | ✅       | Sink                      | ❌              |
| ParticleSinkSubcycle              | 3   | ✅     | ❌   | ❌                 | ❌       | Sink                      | ❌              |
| ParticleStarEvolution             | 3   | ✅     | ✅   | ❌                 | ✅       | Star                      | ❌              |
| PassiveScalar                     | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| PopIII                            | 3   | ✅     | ❌   | ❌                 | ✅       | ❌                         | ❌              |
| PrimordialChem                    | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| RadDust                           | 1   | ✅     | ❌   | SG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadDustMG                         | 1   | ✅     | ❌   | MG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadForce                          | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadLineCooling                    | 1   | ✅     | ❌   | SG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadLineCoolingMG                  | 1   | ✅     | ❌   | MG+ThermalDust+PE | ❌       | ❌                         | ❌              |
| RadMarshak                        | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMarshakAsymptotic              | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMarshakCGS                     | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMarshakDust                    | 1   | ❌     | ❌   | MG+ThermalDust    | ❌       | ❌                         | ❌              |
| RadMarshakDustPE                  | 1   | ❌     | ❌   | MG+ThermalDust+PE | ❌       | ❌                         | ❌              |
| RadMarshakVaytet                  | 1   | ❌     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadMatterCoupling                 | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadMatterCouplingRSLA             | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadStreaming                      | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadStreamingY                     | 2   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadSuOlson                        | 1   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadTube                           | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroBB                        | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroPulse                     | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroPulseDyn                  | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroPulseGrey                 | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroPulseMGconst              | 1   | ✅     | ✅   | MG                | ❌       | ❌                         | ❌              |
| RadhydroPulseMGint                | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroShell                     | 3   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroShock                     | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroShockCGS                  | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RadhydroShockMultigroup           | 1   | ✅     | ❌   | MG                | ❌       | ❌                         | ❌              |
| RadhydroUniformAdvecting          | 1   | ✅     | ❌   | SG                | ❌       | ❌                         | ❌              |
| RandomBlast                       | 3   | ✅     | ❌   | ❌                 | ✅       | StochasticStellarPop      | 1              |
| RayleighTaylor3D                  | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
| ResampledCoolingTest              | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| RyuJones2aShockTube               | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| SN                                | 3   | ✅     | ✅   | ❌                 | ❌       | Test                      | 1              |
| ShockCloud                        | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 3              |
| SlowWaveConvergence               | 3   | ✅     | ✅   | ❌                 | ❌       | ❌                         | ❌              |
| SphericalCollapse                 | 3   | ✅     | ❌   | ❌                 | ✅       | CIC                       | ❌              |
| StarCluster                       | 3   | ✅     | ❌   | ❌                 | ✅       | ❌                         | ❌              |
| StromgrenSphere                   | 3   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| StromgrenSphereRSLA               | 3   | ❌     | ❌   | SG                | ❌       | ❌                         | ❌              |
| TallBoxSf                         | 3   | ✅     | ❌   | ❌                 | ✅       | StochasticStellarPop      | 1              |
| ThermalConduction                 | 1   | ✅     | ❌   | ❌                 | ❌       | ❌                         | ❌              |
| Turbulence                        | 3   | ✅     | ❌   | ❌                 | ❌       | ❌                         | 1              |
