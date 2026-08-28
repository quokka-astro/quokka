# src/ Audit Log

Status legend: `pending`, `reviewed`, `finding`, `partial`

## Documentation Read (for architecture/context)
- `docs/markdown/developer_onboarding.md`
- `docs/markdown/equations.md`
- `docs/markdown/hydro_integrator.md`
- `docs/markdown/mhd_module.md`
- `docs/markdown/dust_module.md`
- `docs/markdown/particles.md`

## File Coverage Checklist
- [x] `src/QuokkaSimulation.cpp` (`reviewed`)
- [x] `src/QuokkaSimulation.hpp` (`reviewed`)
- [x] `src/SimulationData.hpp` (`reviewed`)
- [x] `src/chemistry/Chemistry.cpp` (`reviewed`)
- [x] `src/chemistry/Chemistry.hpp` (`reviewed`)
- [x] `src/cooling/PhotoelectricHeating.hpp` (`reviewed`)
- [x] `src/cooling/ResampledCooling.cpp` (`reviewed`)
- [x] `src/cooling/ResampledCooling.hpp` (`reviewed`)
- [x] `src/dust/DustDrag.cpp` (`reviewed`)
- [x] `src/dust/DustDrag.hpp` (`reviewed`)
- [x] `src/dust/DustState.hpp` (`reviewed`)
- [x] `src/dust/dustRiemannSolver.hpp` (`reviewed`)
- [x] `src/dust/dust_system.cpp` (`reviewed`)
- [x] `src/dust/dust_system.hpp` (`reviewed`)
- [x] `src/grid.hpp` (`reviewed`)
- [x] `src/hydro/EOS.hpp` (`reviewed`)
- [x] `src/hydro/HLLC.hpp` (`reviewed`)
- [x] `src/hydro/HLLD.hpp` (`reviewed`)
- [x] `src/hydro/HydroState.hpp` (`reviewed`)
- [x] `src/hydro/LLF.hpp` (`reviewed`)
- [x] `src/hydro/LLF_mhd.hpp` (`reviewed`)
- [x] `src/hydro/NSCBC_inflow.hpp` (`reviewed`)
- [x] `src/hydro/NSCBC_outflow.hpp` (`reviewed`)
- [x] `src/hydro/hydro_system.cpp` (`reviewed`)
- [x] `src/hydro/hydro_system.hpp` (`reviewed`)
- [x] `src/hydro/mhd_system.cpp` (`reviewed`)
- [x] `src/hydro/mhd_system.hpp` (`reviewed`)
- [x] `src/hyperbolic_system.cpp` (`reviewed`)
- [x] `src/hyperbolic_system.hpp` (`reviewed`)
- [x] `src/io/DiagBase.cpp` (`reviewed`)
- [x] `src/io/DiagFilter.cpp` (`reviewed`)
- [x] `src/io/DiagFramePlane.cpp` (`reviewed`)
- [x] `src/io/DiagPDF.cpp` (`reviewed`)
- [x] `src/io/DiagParticleTxt.cpp` (`reviewed`)
- [x] `src/io/DiagPlotfile.cpp` (`reviewed`)
- [x] `src/io/DiagProjectionPlot.cpp` (`reviewed`)
- [x] `src/io/io_utils.hpp` (`reviewed`)
- [x] `src/io/openPMD.cpp` (`reviewed`)
- [x] `src/io/openPMD.hpp` (`reviewed`)
- [x] `src/io/projection.cpp` (`reviewed`)
- [x] `src/io/projection.hpp` (`reviewed`)
- [x] `src/linear_advection/AdvectionSimulation.cpp` (`reviewed`)
- [x] `src/linear_advection/AdvectionSimulation.hpp` (`reviewed`)
- [x] `src/linear_advection/linear_advection.cpp` (`reviewed`)
- [x] `src/linear_advection/linear_advection.hpp` (`reviewed`)
- [x] `src/main.cpp` (`reviewed`)
- [x] `src/main.hpp` (`reviewed`)
- [x] `src/math/FastMath.hpp` (`reviewed`)
- [x] `src/math/Interpolate2D.cpp` (`reviewed`)
- [x] `src/math/Interpolate2D.hpp` (`reviewed`)
- [x] `src/math/ODEIntegrate.hpp` (`reviewed`)
- [ ] `src/math/gauss.hpp` (`partial`)
- [x] `src/math/interpolate.hpp` (`reviewed`)
- [x] `src/math/math_impl.hpp` (`reviewed`)
- [x] `src/math/quadrature.hpp` (`reviewed`)
- [x] `src/math/root_finding.hpp` (`reviewed`)
- [x] `src/particles/PhysicsParticles.hpp` (`reviewed`)
- [x] `src/particles/particle_IO.hpp` (`reviewed`)
- [x] `src/particles/particle_accretion.hpp` (`reviewed`)
- [x] `src/particles/particle_creation.hpp` (`reviewed`)
- [x] `src/particles/particle_deposition.hpp` (`reviewed`)
- [x] `src/particles/particle_destruction.hpp` (`reviewed`)
- [x] `src/particles/particle_radiation.hpp` (`reviewed`)
- [x] `src/particles/particle_types.hpp` (`reviewed`)
- [x] `src/particles/particle_update.hpp` (`reviewed`)
- [x] `src/particles/particle_utils.hpp` (`reviewed`)
- [x] `src/particles/stellarpop_data.hpp` (`reviewed`)
- [x] `src/physics_info.hpp` (`reviewed`)
- [x] `src/physics_numVars.hpp` (`reviewed`)
- [x] `src/problems/Advection/testAdvection.cpp` (`reviewed`)
- [x] `src/problems/Advection2D/testAdvection2D.cpp` (`reviewed`)
- [x] `src/problems/AdvectionSemiellipse/testAdvectionSemiellipse.cpp` (`reviewed`)
- [x] `src/problems/AlfvenWaveCircular/testAlfvenWaveCircular.cpp` (`reviewed`)
- [x] `src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp` (`reviewed`)
- [x] `src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp` (`reviewed`)
- [x] `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp` (`reviewed`)
- [x] `src/problems/BrioWuShockTube/testBrioWuShockTube.cpp` (`reviewed`)
- [x] `src/problems/Cooling/testCooling.cpp` (`reviewed`)
- [x] `src/problems/CurrentSheet/testCurrentSheet.cpp` (`reviewed`)
- [x] `src/problems/DiskGalaxy/testDiskGalaxy.cpp` (`reviewed`)
- [x] `src/problems/DustAdvection/testDustAdvection.cpp` (`reviewed`)
- [x] `src/problems/DustAdvection3D/testDustAdvection3D.cpp` (`reviewed`)
- [x] `src/problems/DustDamping/testDustDamping.cpp` (`reviewed`)
- [x] `src/problems/DustDampingIteration/testDustDampingIteration.cpp` (`reviewed`)
- [x] `src/problems/DustSoundwave/testDustSoundwave.cpp` (`reviewed`)
- [x] `src/problems/DustyShock/testDustyShock.cpp` (`reviewed`)
- [x] `src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp` (`reviewed`)
- [x] `src/problems/FCQuantities/testFCQuantities.cpp` (`reviewed`)
- [x] `src/problems/FastWave/testFastWave.cpp` (`reviewed`)
- [x] `src/problems/FastWave/test_fast_wave.hpp` (`reviewed`)
- [x] `src/problems/FastWaveConvergence/testFastWaveConvergence.cpp` (`reviewed`)
- [x] `src/problems/FieldLoop/testFieldLoop.cpp` (`reviewed`)
- [x] `src/problems/GravRadParticle3D/testGravRadParticle3D.cpp` (`reviewed`)
- [x] `src/problems/HydroBlast2D/testHydroBlast2D.cpp` (`reviewed`)
- [x] `src/problems/HydroBlast3D/testHydroBlast3D.cpp` (`reviewed`)
- [x] `src/problems/HydroContact/testHydroContact.cpp` (`reviewed`)
- [x] `src/problems/HydroHighMach/testHydroHighMach.cpp` (`reviewed`)
- [x] `src/problems/HydroKelvinHelmholz/testHydroKelvinHelmholz.cpp` (`reviewed`)
- [x] `src/problems/HydroLeblanc/testHydroLeblanc.cpp` (`reviewed`)
- [x] `src/problems/HydroQuirk/testHydroQuirk.cpp` (`reviewed`)
- [x] `src/problems/HydroRichtmeyerMeshkov/testHydroRichtmeyerMeshkov.cpp` (`reviewed`)
- [x] `src/problems/HydroSMS/testHydroSMS.cpp` (`reviewed`)
- [x] `src/problems/HydroShocktube/testHydroShocktube.cpp` (`reviewed`)
- [x] `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp` (`reviewed`)
- [x] `src/problems/HydroShuOsher/testHydroShuOsher.cpp` (`reviewed`)
- [x] `src/problems/HydroVacuum/testHydroVacuum.cpp` (`reviewed`)
- [x] `src/problems/HydroWave/testHydroWave.cpp` (`reviewed`)
- [x] `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` (`reviewed`)
- [x] `src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp` (`reviewed`)
- [x] `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp` (`reviewed`)
- [x] `src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp` (`reviewed`)
- [x] `src/problems/MHDBlast/testMHDBlast.cpp` (`reviewed`)
- [x] `src/problems/MHDQuirk/testMHDQuirk.cpp` (`reviewed`)
- [x] `src/problems/NscbcChannel/testNscbcChannel.cpp` (`reviewed`)
- [x] `src/problems/NscbcVortex/testNscbcVortex.cpp` (`reviewed`)
- [x] `src/problems/ODEIntegration/testODEIntegration.cpp` (`reviewed`)
- [x] `src/problems/OrszagTang/testOrszagTang.cpp` (`reviewed`)
- [x] `src/problems/ParticleAccretion/testParticleAccretion.cpp` (`reviewed`)
- [x] `src/problems/ParticleCreation/testParticleCreation.cpp` (`reviewed`)
- [x] `src/problems/ParticleRadiation/testParticleRadiation.cpp` (`reviewed`)
- [x] `src/problems/ParticleSF/testParticleSF.cpp` (`reviewed`)
- [x] `src/problems/ParticleSink/testParticleSink.cpp` (`reviewed`)
- [x] `src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp` (`reviewed`)
- [x] `src/problems/PassiveScalar/testPassiveScalar.cpp` (`reviewed`)
- [x] `src/problems/PopIII/testPopIII.cpp` (`reviewed`)
- [x] `src/problems/PrimordialChem/testPrimordialChem.cpp` (`reviewed`)
- [x] `src/problems/RadBeam/testRadBeam.cpp` (`reviewed`)
- [x] `src/problems/RadDust/testRadDust.cpp` (`reviewed`)
- [x] `src/problems/RadDustMG/testRadDustMG.cpp` (`reviewed`)
- [x] `src/problems/RadForce/testRadForce.cpp` (`reviewed`)
- [x] `src/problems/RadLineCooling/testRadLineCooling.cpp` (`reviewed`)
- [x] `src/problems/RadLineCoolingMG/testRadLineCoolingMG.cpp` (`reviewed`)
- [x] `src/problems/RadMarshak/testRadMarshak.cpp` (`reviewed`)
- [x] `src/problems/RadMarshakAsymptotic/testRadMarshakAsymptotic.cpp` (`reviewed`)
- [x] `src/problems/RadMarshakCGS/testRadMarshakCGS.cpp` (`reviewed`)
- [x] `src/problems/RadMarshakDust/testRadMarshakDust.cpp` (`reviewed`)
- [x] `src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp` (`reviewed`)
- [x] `src/problems/RadMarshakVaytet/testRadMarshakVaytet.cpp` (`reviewed`)
- [x] `src/problems/RadMatterCoupling/testRadMatterCoupling.cpp` (`reviewed`)
- [x] `src/problems/RadMatterCouplingRSLA/testRadMatterCouplingRSLA.cpp` (`reviewed`)
- [x] `src/problems/RadShadow/testRadShadow.cpp` (`reviewed`)
- [x] `src/problems/RadStreaming/testRadStreaming.cpp` (`reviewed`)
- [x] `src/problems/RadStreamingY/testRadStreamingY.cpp` (`reviewed`)
- [x] `src/problems/RadSuOlson/testRadSuOlson.cpp` (`reviewed`)
- [x] `src/problems/RadTophat/testRadTophat.cpp` (`reviewed`)
- [x] `src/problems/RadTube/testRadTube.cpp` (`reviewed`)
- [x] `src/problems/RadhydroBB/testRadhydroBB.cpp` (`reviewed`)
- [x] `src/problems/RadhydroPulse/testRadhydroPulse.cpp` (`reviewed`)
- [x] `src/problems/RadhydroPulseDyn/testRadhydroPulseDyn.cpp` (`reviewed`)
- [x] `src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp` (`reviewed`)
- [x] `src/problems/RadhydroPulseMGconst/testRadhydroPulseMGconst.cpp` (`reviewed`)
- [x] `src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp` (`reviewed`)
- [x] `src/problems/RadhydroShell/testRadhydroShell.cpp` (`reviewed`)
- [x] `src/problems/RadhydroShock/testRadhydroShock.cpp` (`reviewed`)
- [x] `src/problems/RadhydroShockCGS/testRadhydroShockCGS.cpp` (`reviewed`)
- [x] `src/problems/RadhydroShockMultigroup/testRadhydroShockMultigroup.cpp` (`reviewed`)
- [x] `src/problems/RadhydroUniformAdvecting/testRadhydroUniformAdvecting.cpp` (`reviewed`)
- [x] `src/problems/RandomBlast/testRandomBlast.cpp` (`reviewed`)
- [x] `src/problems/RayleighTaylor2D/testRayleighTaylor2D.cpp` (`reviewed`)
- [x] `src/problems/RayleighTaylor3D/testRayleighTaylor3D.cpp` (`reviewed`)
- [x] `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp` (`reviewed`)
- [x] `src/problems/SN/testSN.cpp` (`reviewed`)
- [x] `src/problems/ShockCloud/testShockCloud.cpp` (`reviewed`)
- [x] `src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp` (`reviewed`)
- [x] `src/problems/SphericalCollapse/testSphericalCollapse.cpp` (`reviewed`)
- [x] `src/problems/StarCluster/testStarCluster.cpp` (`reviewed`)
- [x] `src/problems/TallBoxSf/testTallBoxSf.cpp` (`reviewed`)
- [x] `src/problems/Turbulence/testTurbulence.cpp` (`reviewed`)
- [x] `src/radiation/planck_integral.hpp` (`reviewed`)
- [x] `src/radiation/radiation_dust_system.hpp` (`reviewed`)
- [x] `src/radiation/radiation_system.hpp` (`reviewed`)
- [x] `src/radiation/source_terms_multi_group.hpp` (`reviewed`)
- [ ] `src/radiation/source_terms_single_group.hpp` (`finding`)
- [x] `src/simulation.cpp` (`reviewed`)
- [x] `src/simulation.hpp` (`reviewed`)
- [x] `src/turbulence/TurbDataReader.cpp` (`reviewed`)
- [x] `src/turbulence/TurbDataReader.hpp` (`reviewed`)
- [x] `src/turbulence/TurbulentDriving.cpp` (`reviewed`)
- [x] `src/turbulence/TurbulentDriving.hpp` (`reviewed`)
- [x] `src/util/ArrayUtil.hpp` (`reviewed`)
- [x] `src/util/ArrayView.hpp` (`reviewed`)
- [x] `src/util/ArrayView_2d.hpp` (`reviewed`)
- [x] `src/util/ArrayView_3d.hpp` (`reviewed`)
- [x] `src/util/BC.hpp` (`reviewed`)
- [x] `src/util/CheckNaN.hpp` (`reviewed`)
- [x] `src/util/DataTable.hpp` (`reviewed`)
- [x] `src/util/Optional.hpp` (`reviewed`)
- [x] `src/util/fextract.cpp` (`reviewed`)
- [x] `src/util/fextract.hpp` (`reviewed`)
- [x] `src/util/richardson.hpp` (`reviewed`)
- [x] `src/util/valarray.hpp` (`reviewed`)

## Reviewed Functions (detailed)

### `src/main.cpp`
- `main(int, char**)` (`reviewed`): Entry-point and AMReX initialization path reviewed; no clear correctness bug found in this pass.

### `src/main.hpp`
- `main(int, char**)` declaration (`reviewed`): interface only.
- `problem_main()` declaration (`reviewed`): interface only (implemented by problem generators).

### `src/simulation.cpp`
- No function definitions (`reviewed`). File only includes `simulation.hpp`.

### `src/simulation.hpp`
- `AMRSimulation::initialize()` (`reviewed`): setup/allocation, nesting checks, metadata init, and optional Ascent startup reviewed; no new confirmed bug in inspected section.
- `AMRSimulation::readParameters()` (`reviewed`): runtime parsing, density-floor parser setup, and particle table initialization path reviewed; no new confirmed bug in inspected section.
- `AMRSimulation::rereadRuntimeParameters()` (`reviewed`): thin wrapper over `readParameters()`.
- `AMRSimulation::setInitialConditions()` (`reviewed`): fresh-start/restart init flow, post-init output hooks, diagnostics init, and performance hint path reviewed; no new confirmed bug in inspected section.
- `AMRSimulation::computeTimestepAtLevel(int)` (`finding`): `hydro_dt` is computed as `dx_min / domain_signal_max` without guarding `domain_signal_max <= 0` or non-finite values (`src/simulation.hpp:1171`). A zero/NaN signal speed can produce `inf`/`nan` timestep and contaminate later timestep logic.
- `AMRSimulation::computeTimestep()` (`reviewed`): per-level candidate aggregation, subcycling coupling, initial-step modifiers, and stop-time clipping reviewed; no new confirmed bug in inspected section beyond existing CFL robustness finding in `computeTimestepAtLevel`.
- `AMRSimulation::getWalltime()` (`reviewed`): static-start stopwatch implementation looks consistent.
- `AMRSimulation::getCycleWalltime()` (`reviewed`): per-cycle stopwatch implementation looks consistent.
- `AMRSimulation::evolve()` (`finding`): time-based checkpoint scheduling is initialized with `next_chk_file_time = 0` and not advanced to `checkpointTimeInterval_` on fresh starts (`src/simulation.hpp:1338-1345`), unlike plotfiles (`src/simulation.hpp:1328-1337`). This causes `checkpointtime_interval` to trigger on the first completed step (`src/simulation.hpp:1530-1535`) rather than after one full interval.
- `AMRSimulation::evolve()` (`finding`): time sync assert divides by `cur_time` (`src/simulation.hpp:1459`). If a zero timestep is produced (e.g., pathological CFL calculation), this assertion becomes `0/0` or `x/0` and fails non-diagnostically.
- `AMRSimulation::calculateGpotAllLevels()` (`finding`): the OpenBCSolver branch computes `abstol = abstolPoisson_ * rhs_min` (`src/simulation.hpp:1842`) using the signed minimum RHS value, while the MLMG branch uses `std::abs(rhs_min)` (`src/simulation.hpp:1821`). For the common case `rhs_min < 0`, this can pass a negative absolute tolerance to the solver.
- `AMRSimulation::gravAccelAllLevels(Real)` (`reviewed`): thin self-gravity wrapper over `applyPoissonGravityAtLevel(...)`; no confirmed bug in this wrapper.
- `AMRSimulation::ellipticSolveAllLevels(Real)` (`reviewed`): Poisson supercycling gate + per-step gravity kick logic reviewed; no confirmed bug in inspected logic.
- `setFunctorParticleAccel::operator()(...)` (`reviewed`): GPU boundary functor zero-fills ext_dir `phi` ghost cells for particle acceleration; implementation matches surrounding comments.
- `AMRSimulation::kickParticlesAllLevels(Real)` (`reviewed`): `phi` ghost-fill strategy, two-level fill path, and centered-gradient acceleration assembly reviewed; no confirmed bug in inspected section.
- `AMRSimulation::particleMeshInteraction(Real, Real)` (`reviewed`): sink accretion / particle creation / SN deposition orchestration at the finest level reviewed; no new confirmed bug in this pass.
- `AMRSimulation::timeStepWithSubcycling(int, Real, int)` (`reviewed`): regrid/recursive advance/reflux/average-down/particle-redistribution flow reviewed; no new confirmed bug in inspected section.
- `AMRSimulation::incrementFluxRegisters(...)` (`reviewed`): face-area scaling and coarse/fine flux register updates look consistent across dimensions.
- `AMRSimulation::incrementEMFRegisters(...)` (`reviewed`): 3D edge-flux register accumulation path reviewed; no confirmed bug in inspected section.
- `AMRSimulation::getAmrInterpolaterCellCentered()` (`reviewed`): method-selection logic is straightforward; invalid values abort.
- `AMRSimulation::getAmrInterpolaterFaceCentered()` (`reviewed`): returns `face_linear_interp` for generic face-centered fills.
- `AMRSimulation::MakeNewLevelFromCoarse(...)` (`finding`, correctness): constructs per-direction face BC arrays with `BCs_array[idim] = BCs_fc_` (`src/simulation.hpp:2367`) instead of slicing the flattened `BCs_fc_` vector to `ncomp_per_dim_fc` entries. `FillCoarsePatchFaceArray(...)` then receives incorrect BC vector sizes/ordering.
- `AMRSimulation::RemakeLevel(...)` (`finding`, correctness): `int_state_old_cc` is allocated but never filled before `std::swap(int_state_old_cc, state_old_cc_[level])` (`src/simulation.hpp:2386-2389`), so `state_old_cc_[level]` becomes uninitialized after remaking a level.
- `AMRSimulation::RemakeLevel(...)` (`finding`, correctness): repeats the same face-BC packing bug as `MakeNewLevelFromCoarse` (`BCs_array[idim] = BCs_fc_` at `src/simulation.hpp:2416`), passing flattened BC records where per-direction BC vectors are expected.
- `AMRSimulation::ClearLevel(int)` (`reviewed`): clears MultiFabs/registers/fillpatchers consistently.
- `AMRSimulation::InterpHookNone(...)` (`reviewed`): intentional no-op interpolation hook.
- `setBoundaryFunctor::operator()(...)` (`reviewed`): direct forwarder to problem-specific custom BC callback.
- `setBoundaryFunctorFaceVar::{ctor, operator()}` (`reviewed`): direction-carrying wrapper and face-var custom-BC dispatch reviewed; default `na` direction intentionally becomes a no-op.
- `AMRSimulation::setCustomBoundaryConditions(...)` (`reviewed`): default ext_dir hook is an intentional no-op placeholder (specialization point).
- `AMRSimulation::setCustomBoundaryConditionsFaceVar<...>(...)` (`reviewed`): default face-var ext_dir hook is an intentional no-op placeholder (specialization point).
- `AMRSimulation::setConstantDirichletBCLo<...>(...)` (`reviewed`): lower-boundary constant Dirichlet helper logic is straightforward.
- `AMRSimulation::setConstantDirichletBCHi<...>(...)` (`reviewed`): upper-boundary constant Dirichlet helper logic is straightforward.
- `AMRSimulation::setDiodeBCLo<...>(...)` (`finding`, correctness): diode ghost fill copies/reflects only `{rho, mom, E, Eint}` and leaves additional conserved components (e.g. passive scalars) untouched (`src/simulation.hpp:2642-2653`, `:2689-2694`). This can leave stale ghost values when diode BCs are used with extra hydro state components.
- `AMRSimulation::setDiodeBCHi<...>(...)` (`finding`, correctness): same issue on the upper boundary; only core hydro fields are filled (`src/simulation.hpp:2753-2764`, `:2800-2805`), so passive scalars/other extra conserved components are not boundary-populated.
- `AMRSimulation::setConstantDirichletBCFaceVarLo<...>(...)` (`reviewed`): face-var lower-boundary helper and normal/tangential inclusivity rules reviewed; no confirmed bug in inspected section.
- `AMRSimulation::setConstantDirichletBCFaceVarHi<...>(...)` (`reviewed`): face-var upper-boundary helper reviewed; no confirmed bug in inspected section.
- `AMRSimulation::FillPatch(...)` (`reviewed`): generic AMR fill wrapper (used by `AdvectionSimulation`) mirrors Quokka override behavior.
- `AMRSimulation::setInitialConditionsAtLevel_cc(int, Real)` (`reviewed`): per-grid IC fill, ghost fill, and old/new sync path reviewed.
- `AMRSimulation::setInitialConditionsAtLevel_fc(int, Real)` (`reviewed`): per-face IC fill, overlap sync, ghost fill, and old/new sync path reviewed.
- `AMRSimulation::MakeNewLevelFromScratch(...)` (`reviewed`): initialization-time level allocation and IC setup path reviewed.
- `AMRSimulation::fillBoundaryConditions(...)` (`reviewed`): level-0/refined-level fill orchestration, custom BC dispatch, and NaN assertions reviewed; no confirmed bug in inspected section.
- `AMRSimulation::FillPatchWithData(...)` (`reviewed`): single-/two-level fill paths, `FillPatcher` integration, and cc/fc physical BC functor dispatch reviewed; no confirmed bug in inspected section.
- `AMRSimulation::FillCoarsePatch(...)` (`finding`, correctness): face-centered branch still constructs cell-centered boundary functors (`setBoundaryFunctor`) and never uses `setBoundaryFunctorFaceVar` / `dir` (`src/simulation.hpp:3240-3250`), so custom face-variable physical BCs are skipped on coarse interpolation fills.
- `AMRSimulation::FillCoarsePatchFaceArray(...)` (`reviewed`): simultaneous face-array coarse interpolation path reviewed; no confirmed bug in inspected section.
- `AMRSimulation::GetData(...)` (`reviewed`): completed review of old/new/both time-selection branches for cc/fc data pointers.
- `AMRSimulation::GetDataFaceArray(...)` (`reviewed`): face-array time-selection helper mirrors `GetData(...)` logic across all directions.
- `AMRSimulation::AverageDown()` (`reviewed`): coarse-level averaging loop is straightforward.
- `AMRSimulation::AverageDownTo(int)` (`reviewed`): cell- and face-centered averaging-down paths reviewed; no confirmed bug in inspected section.
- `AMRSimulation::computeVolumeIntegral(F)` (`reviewed`): per-level evaluation + AMReX `volumeWeightedSum` path reviewed.
- `AMRSimulation::InitParticles()` (`reviewed`): tracer particle container init path is straightforward.
- `AMRSimulation::InitPhyParticles(...)` (`reviewed`): particle-container initialization/restart registration logic through all enabled particle types reviewed; no new confirmed bug in this pass.
- `AMRSimulation::PlotFileName(int) const` (`reviewed`): simple formatter wrapper.
- `AMRSimulation::CustomPlotFileName(const char*, int) const` (`reviewed`): simple formatter wrapper.
- `AMRSimulation::AverageFCToCC(...) const` (`reviewed`): face-to-cell averaging helper reviewed; dimension offsets and ghost requirements are explicit.
- `AMRSimulation::PlotFileMFAtLevel_cc(...)` (`reviewed`): configurable cell/face/derived plot variable assembly path reviewed; no new confirmed bug in inspected section.
- `AMRSimulation::ComputeDensityFloorDebug(...) const` (`reviewed`): density-floor debug field fill (constant/parser) reviewed.
- `AMRSimulation::PlotFileMFAtLevel_fc(...)` (`reviewed`): face-centered plot MultiFab assembly path is straightforward.
- `AMRSimulation::AverageDownDerived(...) const` (`reviewed`): derived-component-only averaging-down helper reviewed.
- `AMRSimulation::PlotFileMF_cc(...)` (`reviewed`): per-level plot MultiFab collection + derived averaging + optional periodic ghost fill reviewed.
- `AMRSimulation::PlotFileMF_fc(...)` (`reviewed`): per-dimension/per-level face plot MultiFab collection is straightforward.
- `AMRSimulation::createDiagnostics()` (`reviewed`): diagnostic list parsing/broadcast, factory creation, var validation, and `DiagPlotfile` regular-output suppression reviewed; no new confirmed bug in inspected section.
- `AMRSimulation::updateDiagnostics()` (`reviewed`): refreshes diagnostics that require grid-change updates.
- `AMRSimulation::doDiagnostics()` (`reviewed`): completed review of diagnostic scheduling, data assembly, averaging, and dynamic dispatch to supported diagnostic types.
- `AMRSimulation::AscentCustomActions(...)` (`reviewed`): default Ascent pseudocolor scene/action setup reviewed (compile-time optional).
- `AMRSimulation::RenderAscent()` (`reviewed`): Ascent render path (plot-MF assembly, convexify, geometry rescale, blueprint export) reviewed; no confirmed bug in inspected section.
- `AMRSimulation::GetPlotfileVarNames() const` (`reviewed`): returns configured CC plot variable list.
- `AMRSimulation::GetPlotfileVarNames_fc() const` (`reviewed`): builds per-dimension FC plot variable names from `componentNames_fc_`.
- `AMRSimulation::WritePlotFile()` (`reviewed`): plotfile/openPMD output path, FC plotfile sidecar writes, metadata writes, and particle output reviewed; no new confirmed bug in this pass (OpenPMD field-filtering issue is logged in `src/io/DiagPlotfile.H`).
- `AMRSimulation::WriteMetadataFile(...) const` (`reviewed`): YAML metadata writer reviewed.
- `AMRSimulation::ReadMetadataFile(...)` (`finding`, robustness): uses `feholdexcept(...)` / `fesetenv(...)` for temporary FPE masking (`src/simulation.hpp:4164-4165`, `:4192`), but `YAML::LoadFile(...)` or subsequent parsing can throw before `fesetenv(...)` runs, leaving the process FPE environment altered.
- `AMRSimulation::WriteStatisticsFile()` (`reviewed`): statistics append/header logic reviewed; no confirmed bug in inspected section.
- `AMRSimulation::SetLastCheckpointSymlink(...) const` (`reviewed`): updates `last_chk` symlink on IO rank.
- `AMRSimulation::WriteCheckpointFile() const` (`reviewed`): checkpoint header/data/metadata/particle write path reviewed; no new confirmed bug in inspected section.
- `GotoNextLine(std::istream&)` (`reviewed`): simple header-parse helper.
- `AMRSimulation::detectRefinementContext(...)` (`reviewed`): restart-refinement detection and coarse level-0 geometry construction reviewed.
- `AMRSimulation::readCheckpointHeader(...)` (`finding`, robustness): parses `istep`, `dt_`, and `tNew_` lines with unbounded `while (lis >> word) { arr[i++] = ...; }` loops (`src/simulation.hpp:4437-4440`, `:4447-4450`, `:4457-4460`) into fixed-size arrays, so malformed headers with extra tokens can overflow.
- `AMRSimulation::interpolateMultiFabFromRestart(...)` (`reviewed`): restart cell-centered copy/refine interpolation helper reviewed.
- `AMRSimulation::interpolateFaceMultiFabFromRestart(...)` (`reviewed`): refined/non-refined face restart interpolation orchestration (including face-divfree interpolation path) reviewed; no confirmed bug in inspected section.
- `AMRSimulation::loadMultiFabData(...)` (`reviewed`): restart MultiFab loading and refinement-aware face-data staging path reviewed; no confirmed bug in inspected section.
- `AMRSimulation::loadBalanceOnRestart(...)` (`reviewed`): restart BoxArray load-balancing helper reviewed.
- `AMRSimulation::ReadCheckpointFile()` (`reviewed`): restart flow through header parse, refinement context, grid setup, metadata/data load, and particle restart init reviewed.
- `AMRSimulation::restartParticleContainerWithRefinement(...)` (`reviewed`): checkpoint particle presence checks, coarse-geometry restart workaround, and redistribution path reviewed.
- `AMRSimulation::initializeParticleContainerFromCheckpoint<...>(...)` (`reviewed`): container create/register/restart/split-on-refine wrapper reviewed.
- `AMRSimulation::writeFaceVelocitiesToDisk(...)` (`reviewed`): debug ASCII dump helper reviewed.
- `AMRSimulation::writeReconstructedStatesToDisk(...)` (`reviewed`): debug ASCII dump helper for reconstructed interface states reviewed.

### `src/QuokkaSimulation.hpp`
- `QuokkaSimulation::readParmParse()` (`reviewed`): reviewed hydro/MHD/cooling/turbulence/radiation parameter parsing and PE-heating table setup/validation path; no new confirmed correctness bug in inspected sections.
- `QuokkaSimulation::rereadRuntimeParameters()` (`reviewed`): parent reread + Quokka-specific reread + particle reread; behavior consistent with `evolve()` re-override design.
- `QuokkaSimulation::computeNumberOfRadiationSubsteps(int, Real)` (`finding`, robustness): computes `dtrad_tmp = radiationCflNumber_ * (dx_min / c_hat)` and `ceil(dt_lev_hydro / dtrad_tmp)` with no guard for `c_hat <= 0` or non-positive `radiationCflNumber_` (`src/QuokkaSimulation.hpp:718-721`), allowing division by zero / invalid substep counts from bad runtime parameters.
- `QuokkaSimulation::computeMaxSignalLocal(int)` (`reviewed`): main signal-speed assembly logic reviewed; portability finding for hard-coded `idim < 3` loop is logged below.
- `QuokkaSimulation::printCellProperties(int, IntVect)` (`reviewed`): diagnostic path reviewed; can emit NaNs if called on invalid states, but no core correctness bug in the helper itself.
- `QuokkaSimulation::CheckHydroStates(...)` / `checkHydroStates(...)` (`reviewed`): debug-only checkpoint + abort path looks consistent.
- `QuokkaSimulation::preCalculateInitialConditions()` (`reviewed`): default no-op hook.
- `QuokkaSimulation::setInitialConditionsOnGrid(...)` (`reviewed`): default no-op hook.
- `QuokkaSimulation::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): default no-op hook.
- `QuokkaSimulation::createInitialRadParticles()` (`reviewed`): default no-op hook.
- `QuokkaSimulation::createInitialCICParticles()` (`reviewed`): default no-op hook.
- `QuokkaSimulation::createInitialCICRadParticles()` (`reviewed`): default no-op hook.
- `QuokkaSimulation::createInitialStochasticStellarPopParticles()` (`reviewed`): default optional no-op hook.
- `QuokkaSimulation::createInitialSinkParticles()` (`reviewed`): default optional no-op hook.
- `QuokkaSimulation::createInitialTestParticles()` (`reviewed`): default optional no-op hook.
- `QuokkaSimulation::computeBeforeTimestep()` (`reviewed`): default no-op hook.
- `QuokkaSimulation::computeAfterTimestep()` (`reviewed`): default no-op hook.
- `QuokkaSimulation::computeAfterLevelAdvance(...)` (`reviewed`): default no-op hook.
- `QuokkaSimulation::addStrangSplitSources(...)` (`reviewed`): default user hook (no-op).
- `QuokkaSimulation::computePhotoelectricHeatingRate(Real)` (`reviewed`): table/null checks and constant-SFR vs particle-SFH dispatch reviewed; no new confirmed bug in inspected section (downstream PE-heating area divide issue is logged in cooling helpers).
- `QuokkaSimulation::addStrangSplitSourcesWithBuiltin(...)` (`reviewed`): cooling/chemistry/turbulence/dust/source chaining and success propagation reviewed; no confirmed bug in inspected section.
- `QuokkaSimulation::ComputeDerivedVar(...)` (`reviewed`): default no-op hook.
- `QuokkaSimulation::ComputeDensityFloorDebug(...)` (`reviewed`): parser/non-parser debug field fill and device-safe captures reviewed; no confirmed bug in inspected section.
- `QuokkaSimulation::advanceHydroAtLevelWithRetries(...)` (`reviewed`): completed review of retry/substep bookkeeping, accepted-state snapshots, restore/update helpers, retry escalation, and fatal debug dump path; no new confirmed bug in inspected implementation.
- `QuokkaSimulation::isCflViolated(...)` (`finding`, robustness): computes `dt_cfl = cflNumber_ * (dx_min / max_signal)` without guarding zero/non-finite `max_signal` (`src/QuokkaSimulation.hpp:2037`), allowing `inf`/`nan` CFL thresholds and unreliable retry acceptance.
- `QuokkaSimulation::advanceHydroAtLevel(...)` (`finding`): Stage-2 face-centered ghost fill uses `time` instead of `time + dt_lev` (`src/QuokkaSimulation.hpp:2309`) while the cell-centered fill uses `time + dt_lev` (`src/QuokkaSimulation.hpp:2305`). This can apply inconsistent boundary states for time-dependent MHD boundary conditions during the RK2 corrector stage.
- `QuokkaSimulation::computeMaxSignalLocal(int)` (`finding`, portability): MHD face arrays are filled with `for (int idim = 0; idim < 3; ++idim)` (`src/QuokkaSimulation.hpp:735`) even though the container type is `std::array<..., AMREX_SPACEDIM>`. This is out-of-bounds for 1D/2D MHD builds.
- `QuokkaSimulation::replaceEMFs(...)` (`finding`, portability): loops `for (int iedge = 0; iedge < 3; ++iedge)` over `std::array<amrex::MultiFab, AMREX_SPACEDIM>` (`src/QuokkaSimulation.hpp:2507`), which is out-of-bounds for 1D/2D MHD builds.
- `QuokkaSimulation::ComputeStatistics()` (`reviewed`): default hook returns empty statistics map.
- `QuokkaSimulation::refineGrid(...)` (`reviewed`): default no-op refinement hook.
- `QuokkaSimulation::ErrorEst(...)` (`reviewed`): wrapper calls user `refineGrid(...)` plus particle-based refinement tagging (3D only).
- `QuokkaSimulation::computeReferenceSolution(...)` (`reviewed`): default no-op reference-solution hook.
- `QuokkaSimulation::computeReferenceSolution_fc(...)` (`reviewed`): default no-op face-centered reference hook.
- `QuokkaSimulation::print_multifab_fc(...)` (`reviewed`): debug-only GPU `printf` helper; noisy but intentionally diagnostic.
- `QuokkaSimulation::densityFloor(...) const` (`reviewed`): default constant floor hook.
- `QuokkaSimulation::computeComponentErrors()` (`reviewed`): reviewed cell-/face-centered reference comparison and reporting logic; no new confirmed bug in this pass.
- `QuokkaSimulation::computeErrorNorm(bool)` (`reviewed`): aggregates component errors into RMS L1-like norm; zero-reference fallback behavior is explicit.
- `QuokkaSimulation::computeAfterEvolve(Vector<Real>&)` (`reviewed`): conservation/summary diagnostics reviewed; no new confirmed bug in this pass.
- `QuokkaSimulation::advanceSingleTimestepAtLevel(int, Real, Real, int)` (`reviewed`): state swap, hydro/radiation/operator-split orchestration, reflux-register reset, and post-step validity checks reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::fillPoissonRhsAtLevel(MultiFab&, int)` (`reviewed`): accumulates hydro density contribution into Poisson RHS (compatible with later particle deposition accumulation).
- `QuokkaSimulation::applyPoissonGravityAtLevel(...)` (`reviewed`): 3D operator-split gravity kick and kinetic-energy correction path reviewed; implementation is explicitly 3D-gated.
- `QuokkaSimulation::projectFaceCenteredMagneticField()` (`reviewed`): projection setup, BC translation, divergence normalization, and retry loop reviewed; no new confirmed bug in inspected section beyond existing MHD portability findings elsewhere.
- `QuokkaSimulation::updateInitialMagneticEnergyFromFaceField()` (`reviewed`): recomputes cell-centered magnetic energy from face fields and syncs total energy; no confirmed bug in inspected section.
- `QuokkaSimulation::postInitialization()` (`reviewed`): MHD post-init projection/energy update + `state_old_fc_` sync path reviewed.
- `QuokkaSimulation::FixupState(int)` (`reviewed`): floor enforcement + dual-energy sync wrapper reviewed.
- `QuokkaSimulation::FillPatch(...)` (`reviewed`): wrapper around `GetData` + `FillPatchWithData` for cc/fc centering.
- `QuokkaSimulation::PreInterpState(...)` (`reviewed`): total-energy to specific-internal-energy transform reviewed; correctness depends on positive-density invariant enforced by upstream hydro fixup.
- `QuokkaSimulation::PostInterpState(...)` (`reviewed`): inverse transform back to total energy reviewed; correctness depends on positive-density invariant enforced by upstream hydro fixup.
- `QuokkaSimulation::computeAxisAlignedProfile(...)` (`reviewed`): evaluates user functor on all levels, averages down, and normalizes line profile; no confirmed bug in inspected section.
- `QuokkaSimulation::printCoordinates(int, IntVect)` (`reviewed`): diagnostic coordinate formatter is straightforward.
- `QuokkaSimulation::replaceFluxes(...)` (`reviewed`): redo-flag-based local first-order flux replacement logic reviewed; no confirmed bug in inspected section.
- `QuokkaSimulation::addFluxArrays(...)` (`reviewed`): component-offset flux accumulation helper is straightforward.
- `QuokkaSimulation::expandFluxArrays(...)` (`reviewed`): reflux-component expansion helper logic reviewed; no confirmed bug in inspected section.
- `QuokkaSimulation::computeHydroFluxes(...)` (`reviewed`): allocation/reconstruction/flattening/flux assembly, debug-output path, and returned face-velocity / MHD-wavespeed buffers reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::computeCCPerpBfieldComps<DIR>(...)` (`reviewed`): perpendicular face-B averaging to cell centers is dimensionally consistent in inspected logic.
- `QuokkaSimulation::hydroFluxFunction<DIR>(...)` (`reviewed`): reconstruction-order dispatch, shock flattening, and HLLC/HLLD flux dispatch reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::computeFOHydroFluxes(...)` (`reviewed`): first-order fallback flux assembly path reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::hydroFOFluxFunction<DIR>(...)` (`reviewed`): completed first-order donor-cell + LLF/LLF_MHD flux dispatch path review.
- `QuokkaSimulation::swapRadiationState(MultiFab&, const MultiFab&)` (`reviewed`): copies radiation-state subset only; behavior is explicit and intentional.
- `QuokkaSimulation::subcycleRadiationAtLevel(...)` (`reviewed`): substep sizing, IMEX stage orchestration, particle radiation deposition, source-term coupling, iteration counters, and failure handling reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::advanceRadiationForwardEuler(...)` (`reviewed`): stage-1 radiation hyperbolic update and reflux-flux buffering path reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::advanceRadiationMidpointRK2(...)` (`reviewed`): stage-2 radiation midpoint update and reflux-flux buffering path reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::computeRadiationFluxes(...)` (`reviewed`): allocates directional flux/diffusive flux FABs and dispatches per-direction `fluxFunction`; no confirmed bug in inspected section.
- `QuokkaSimulation::fluxFunction<DIR>(...)` (`reviewed`): primitive conversion, reconstruction-order dispatch, and radiation flux/diffusive-flux call reviewed; no new confirmed bug in inspected section.
- `QuokkaSimulation::WriteSingleLevelPlotfileSimplified(...)` (`reviewed`): interval-gated debug plotfile wrapper is straightforward.

### `src/hydro/hydro_system.hpp`
- `HydroSystem::ConservedToPrimitive(...)` (`reviewed`): conversion path including MHD magnetic-energy reconstruction and dust primitive conversion reviewed; no new confirmed bug in this pass.
- `HydroSystem::maxSignalSpeedLocal(...)` (`reviewed`): local reduction path reviewed; no new confirmed bug in this pass.
- `HydroSystem::ComputeMaxSignalSpeed(...)` (`finding`, portability): MHD branch unconditionally reads `cons_fc[1]` and `cons_fc[2]` (`src/hydro/hydro_system.hpp:409-412`) from a `std::array<..., AMREX_SPACEDIM>`, causing out-of-bounds access for 1D/2D MHD builds.
- `HydroSystem::CheckStatesValid(...)` (`reviewed`): validation reduction over density/pressure positivity and diagnostic print path reviewed; no confirmed bug in inspected section.
- `HydroSystem::ComputeRhsFromFluxes(...)` (`reviewed`): divergence/sign convention implementation matches documented left-edge flux convention.
- `HydroSystem::PredictStep(...)` (`reviewed`): explicit update + redo flagging looks consistent.
- `HydroSystem::AddFluxesRK2(...)` (`reviewed`): SSPRK2 combine step + redo flagging looks consistent.
- `HydroSystem::ComputeFlatteningCoefficients<DIR>(...)` (`reviewed`): Miller-Colella flattening implementation reviewed; no confirmed bug in inspected section.
- `HydroSystem::FlattenShocks<DIR>(...)` (`reviewed`): shock flattening coefficient application and directional reindexing logic reviewed; no confirmed bug in inspected section.
- `HydroSystem::EnforceLimits(...)` (`reviewed`): density/scalar/dust/temperature floor enforcement path reviewed; no confirmed bug in inspected section.
- `HydroSystem::AddInternalEnergyPdV(...)` (`reviewed`): `P dV` source construction and redo fallback divergence estimate reviewed; no new confirmed bug in this pass.
- `HydroSystem::SyncDualEnergy(...)` (`reviewed`): dual-energy sync logic reviewed; no new confirmed bug in this pass.
- `HydroSystem::ComputeMagneticEnergy(...)` (`finding`, portability): MHD path unconditionally reads `(*cons_fc)[1]` and `(*cons_fc)[2]` (`src/hydro/hydro_system.hpp:668-671`) from a `std::array<..., AMREX_SPACEDIM>`, so it is out-of-bounds for 1D/2D MHD builds.
- `HydroSystem::ComputeFluxes<...>(...)` (`finding`): face-centered normal velocity uses the opposite-side density when deriving `v_norm` from mass flux (`F[rho] >= 0` divides by `rho_R`, `F[rho] < 0` divides by `rho_L`) at `src/hydro/hydro_system.hpp:1514-1521`. This is inconsistent with the immediately following species upwind logic (`src/hydro/hydro_system.hpp:1525-1544`) and with the linear advection implementation (`src/linear_advection/linear_advection.hpp:190-206`), and can bias tracer advection / dual-energy `div v`.

### `src/hydro/mhd_system.hpp`
- `MHDSystem::ComputeEMF(...)` (`reviewed`): dispatch logic over EMF compute schemes is straightforward.
- `MHDSystem::AverageEMF(...)` (`reviewed`): averaging-scheme dispatch logic is straightforward.
- `MHDSystem::ComputeEMF_FelkerStone2017(...)` (`finding`, portability): hard-codes 3D indexing (`fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:179-186` and `:338-340`, so it is not safe for 1D/2D MHD builds.
- `MHDSystem::ComputeEMF_Quokka2026(...)` (`finding`, portability): hard-codes 3D indexing (`fcx_mf_vel[2]`, `fcx_mf_cVars[2]`, `fcx_mf_fspds[2]`, `iedge < 3`) at `src/hydro/mhd_system.hpp:371-385` and `:475-477`, causing out-of-bounds access in 1D/2D MHD builds.
- `MHDSystem::ComputeEMF_Balsara2025(...)` (`finding`, portability): hard-codes 3D storage/loops (`cc_mf_EMF(...,3,...)`, `fcx_mf_cVars[2]`, `idim < 3`, `iedge < 3`) at `src/hydro/mhd_system.hpp:501-503`, `:514-523`, `:532`, and `:571`, so the implementation is not dimension-safe for 1D/2D MHD builds.
- `MHDSystem::EMFAverage_BalsaraSpicer2004(...)` (`reviewed`): quadrant averaging implementation is straightforward.
- `MHDSystem::EMFAverage_LondrilloDelZanna2004(...)` (`reviewed`): weighted EMF averaging reviewed; no confirmed bug in inspected section.
- `MHDSystem::EMFAverage_Balsara2025(...)` (`reviewed`): star-state EMF averaging reviewed; no confirmed bug in inspected section.
- `MHDSystem::ReconstructTo(...)` (`reviewed`): reconstruction-order dispatch logic looks consistent.
- `MHDSystem::SolveInductionEqn(...)` (`finding`, portability): loops over `w0 = 0..2` unconditionally (`src/hydro/mhd_system.hpp:957`) while operating on `std::array<..., AMREX_SPACEDIM>` containers. This is out-of-bounds for 1D/2D MHD builds and should use `AMREX_SPACEDIM`.

### `src/radiation/radiation_system.hpp`
- `RadSystem::ConservedToPrimitive(...)` (`reviewed`): primitive conversion and reduced-flux calculation reviewed; no new confirmed bug in this pass.
- `RadSystem::ComputeMaxSignalSpeed(...)` (`reviewed`): returns uniform `c_hat_` as signal speed.
- `RadSystem::isStateValid(...)` (`reviewed`): positivity/causality check logic is straightforward.
- `RadSystem::amendRadState(...)` (`finding`): comment says NaN `E_r` is handled, but the floor check uses only `if (E_r < Erad_floor_)` (`src/radiation/radiation_system.hpp:695-701`), which is false for NaN. NaN radiation states can therefore survive `amendRadState()` and trip the subsequent assertion in `PredictStep()` (`src/radiation/radiation_system.hpp:756-760`).
- `RadSystem::PredictStep(...)` (`reviewed`): explicit update + amend/assert path reviewed; no additional confirmed bug in inspected section beyond `amendRadState`.
- `RadSystem::AddFluxesRK2(...)` (`reviewed`): IMEX RK2 explicit hyperbolic combine step and dimensional guards reviewed; no confirmed bug in inspected section.
- `RadSystem::ComputeFluxes<DIR>(...)` (`reviewed`): HLL radiation flux assembly, first-order fallback, optional wavespeed correction, and diffusive-flux companion output reviewed; no new confirmed bug in inspected section.
- `RadSystem::ComputeCellOpticalDepth<DIR>(...)` (`finding`, robustness): harmonic-mean optical depth uses `2*tau_L*tau_R/(tau_L+tau_R)` with no zero-denominator guard (`src/radiation/radiation_system.hpp:914-923`). If both sides are optically thin with zero opacity/depth, this becomes `0/0` and can inject NaNs into the optional wavespeed-correction path.
- `RadSystem::ComputeEddingtonTensor(...)` (`reviewed`): M1 Eddington tensor assembly reviewed; no confirmed bug in inspected section.
- `RadSystem::ComputeRadPressure<DIR>(...)` (`finding`, robustness): NaN checks use `AMREX_ASSERT(Fn != NAN)` / `Tn* != NAN` (`src/radiation/radiation_system.hpp:1026-1029`), which are ineffective because comparisons with `NaN` are always true in IEEE arithmetic.
- `RadSystem::ComputePlanckEnergyFractions(...)` (`reviewed`): multigroup Planck energy partition helper reviewed.
- `RadSystem::ComputeNumberDensityH(...)` (`reviewed`): default number-density helper is straightforward.
- `RadSystem::ComputeThermalRadiationSingleGroup(...)` (`reviewed`): thermal radiation scalar + floor reviewed.
- `RadSystem::ComputeThermalRadiationMultiGroup(...)` (`reviewed`): multigroup thermal radiation + per-group floors reviewed.
- `RadSystem::ComputeThermalRadiationTempDerivativeSingleGroup(...)` (`reviewed`): straightforward derivative helper.
- `RadSystem::ComputeThermalRadiationTempDerivativeMultiGroup(...)` (`reviewed`): multigroup derivative helper reviewed.
- `RadSystem::DefineBackgroundHeatingRate(...)` (`reviewed`): default no-op hook (returns zero).
- `RadSystem::DefineNetCoolingRate(...)` (`reviewed`): default no-op cooling hook (returns zeros).
- `RadSystem::DefineNetCoolingRateTempDerivative(...)` (`reviewed`): default no-op derivative hook (returns zeros).
- `RadSystem::DefineCosmicRayHeatingRate(...)` (`reviewed`): default no-op hook (returns zero).
- `RadSystem::SolveLinearEqs(...)` (`reviewed`): structured Jacobian solver helper reviewed.
- `RadSystem::Solve3x3matrix(...)` (`reviewed`): 3x3 elimination helper reviewed.
- `RadSystem::SetRadEnergySource(...)` (`reviewed`): default no-op source hook.
- `RadSystem::ComputeEddingtonFactor(...)` (`reviewed`): Levermore closure helper reviewed.
- `RadSystem::ComputeMassScalars(...)` (`reviewed`): array extraction helper reviewed.
- `RadSystem::ComputePlanckOpacity(...)` (`reviewed`): default stub returns `NaN` (requires problem specialization).
- `RadSystem::ComputeFluxMeanOpacity(...)` (`reviewed`): default dispatches to `ComputePlanckOpacity(...)`.
- `RadSystem::ComputeEnergyMeanOpacity(...)` (`reviewed`): default dispatches to `ComputePlanckOpacity(...)`.
- `RadSystem::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): default stub returns `NaN` tables (requires specialization for multigroup opacity models).
- `RadSystem::ComputeRadQuantityExponents(...)` (`reviewed`): exponent/slope construction, sign-change handling, and optional free-slope normalization path reviewed; no confirmed bug in inspected section.
- `RadSystem::ComputeGroupMeanOpacity(...)` (`reviewed`): group-mean opacity helper reviewed; no confirmed bug in inspected section.
- `RadSystem::ComputeEintFromEgas(...)` (`reviewed`): gas-internal-energy extraction helper reviewed (expects positive density).
- `RadSystem::ComputeEgasFromEint(...)` (`reviewed`): total-gas-energy reconstruction helper reviewed.
- `RadSystem::PlanckFunction(...)` (`reviewed`): Planck integrand helper with large/small-x guards reviewed.
- `RadSystem::ComputeDiffusionFluxMeanOpacity(...)` (`reviewed`): diffusion-limit mean-opacity helper reviewed; denominator guard present.
- `RadSystem::ComputeBinCenterOpacity(...)` (`reviewed`): bin-center opacity helper reviewed.
- `RadSystem::ComputeFluxInDiffusionLimit(...)` (`reviewed`): diffusion-limit multigroup flux helper reviewed.
- `RadSystem::BackwardEulerOneVariable(...)` (`reviewed`): Newton helper reviewed (residual/step convergence tests, max-iteration fail path); no additional confirmed bug in inspected section.
- `RadSystem::ComputeDustTemperatureBateKeto(...)` (`finding`, robustness): warm-start branch computes `T_d = T_gas - R_sum / (N_d * sqrt(T_gas))` with no guard on `N_d > 0` and `T_gas > 0` (`src/radiation/radiation_system.hpp:1480-1483`), so low-density/zero-temperature states can generate `inf`/`nan`.

### `src/radiation/source_terms_multi_group.hpp`
- `RadSystem::ComputeModelDependentKappaEAndKappaP(...)` (`reviewed`): opacity-model dispatch and `kappaP/kappaE/kappaPoverE` assembly reviewed; no confirmed bug in inspected section.
- `RadSystem::ComputeModelDependentKappaFAndDeltaTerms(...)` (`reviewed`): reviewed Planck-edge delta terms and `kappaF` model dispatch; no confirmed bug in this pass.
- `RadSystem::ComputeJacobianForGas(...)` (`reviewed`): multigroup gas-energy Jacobian assembly helper reviewed.
- `RadSystem::SolveGasRadiationEnergyExchange(...)` (`reviewed`): reviewed Newton iteration structure, opacity updates, Jacobian solve, convergence checks, and result packing; no new confirmed bug in inspected section.
- `RadSystem::UpdateFlux(...)` (`reviewed`): reviewed multigroup radiation-flux/gas-momentum update and work-term handling; no new confirmed bug in inspected section.
- `RadSystem::AddSourceTermsMultiGroup(...)` (`reviewed`): reviewed outer iteration orchestration, energy/flux update calls, convergence logic, and state writeback; no new confirmed bug in this pass.

### `src/radiation/source_terms_single_group.hpp`
- `RadSystem::AddSourceTermsSingleGroup(...)` (`finding`): in the `beta_order_ > 1` branch, `gasVel` is value-initialized but never populated before building the 3x3 flux-update matrix (`src/radiation/source_terms_single_group.hpp:399`, `:453-463`). This silently drops the intended velocity-coupling terms in that branch.
- `RadSystem::AddSourceTermsSingleGroup(...)` (`reviewed`): reviewed Newton solve, outer iteration/work-term loop, flux update, and state writeback; no additional confirmed bug in this pass.

### `src/dust/DustDrag.cpp`
- No function definitions (`reviewed`). File only includes `DustDrag.hpp`.

### `src/dust/DustDrag.hpp`
- `DustDrag::ComputeReciprocalStoppingTime(...)` (`reviewed`): default stub returns zeros (requires problem specialization for physical drag).
- `DustDrag::ComputeReciprocalStoppingTimeKwok(...)` (`reviewed`): reviewed Kwok stopping-time helper and guards for non-positive inputs; no confirmed bug in inspected implementation.
- `DustDrag::computeDustDrag(...)` (`reviewed`): reviewed full Picard/update implementation, convergence loop, and momentum/energy writeback path; no confirmed correctness bug in inspected implementation (numerical-method accuracy would still benefit from dedicated regression checks).

### `src/dust/DustState.hpp`
- `quokka::DustState` (`reviewed`): POD state carrier only.

### `src/dust/dustRiemannSolver.hpp`
- `quokka::Riemann::dustRiemannSolver(...)` (`reviewed`): branch logic matches documented Huang & Bai-style advection flux cases.

### `src/dust/dust_system.cpp`
- No function definitions (`reviewed`). File only includes `dust_system.hpp`.

### `src/dust/dust_system.hpp`
- `DustSystem::ComputeDustFluxes<DIR>(...)` (`reviewed`): reviewed dust flux assembly, direction permutation, and flux writeback; no confirmed bug in inspected implementation.

### `src/particles/particle_utils.hpp`
- `ParticleUtils::computePlasmaBeta(...)` (`reviewed`): straightforward helper; zero-magnetic-energy guard present.
- `ParticleUtils::roundoffMultiFab(...)` (`reviewed`): reviewed roundoff/truncation implementation and count-based precision logic; no confirmed bug in inspected implementation.

### `src/particles/particle_types.hpp`
- Global particle parameter definitions (`reviewed`): reviewed runtime globals and parser section.
- `quokka::particleParmParse()` (`finding`): `particle_param3` is declared (`src/particles/particle_types.hpp:471`) but not parsed from inputs; the parser only reads `param1` and `param2` (`src/particles/particle_types.hpp:542-543`), so `particles.param3` is silently ignored.

### `src/particles/particle_update.hpp`
- `ParticlePropertyUpdateTraits<particleType>::updateProperties(...)` (`reviewed`): default no-op specialization hook.
- `ParticlePropertyUpdateTraits<particleType>::updateParticleProperties(...)` (`reviewed`): default no-op container hook.
- `ParticlePropertyUpdateTraits<StochasticStellarPop>::updateProperties(...)` (`reviewed`): luminosity table update dispatch looks consistent.
- `ParticlePropertyUpdateTraits<StochasticStellarPop>::updateParticleProperties(...)` (`reviewed`): container iteration + GPU update path reviewed; no confirmed bug in inspected implementation.

### `src/particles/particle_destruction.hpp`
- `ParticleDestructionImpl::destroyParticlesImpl(...)` (`reviewed`): reviewed mark-invalid + redistribute workflow; known AMR subcycling limitation is documented in-code, no additional confirmed bug in inspected path.
- `ParticleDestructionTraits<particleType>::ParticleChecker::operator()(...)` (`reviewed`): default checker removes particles marked `Removed`.
- `ParticleDestructionTraits<particleType>::destroyParticles(...)` (`reviewed`): thin wrapper over common implementation.

### `src/particles/particle_radiation.hpp`
- `LuminosityTables::const_tables()` (`reviewed`): straightforward host->GPU const-view wrapper.
- `LuminosityTables::is_initialized()` (`reviewed`): straightforward pass-through.
- `LuminosityUpdate::updateLuminosity(...)` (`reviewed`): table interpolation/update logic reviewed; no confirmed bug in inspected implementation.

### `src/particles/particle_deposition.hpp`
- `amrex::ParticleInterpolator::NearestEight` (`reviewed`): custom nearest-eight-cell interpolation weights/index setup reviewed.
- `quokka::RadDeposition::operator()(...)` (`reviewed`): radiation deposition functor reviewed; `AMREX_D_TERM(dxi[0], *dxi[1], *dxi[2])` usage is valid AMReX macro multiplication-token syntax.
- `quokka::MassDeposition::operator()(...)` (`reviewed`): 3D particle-mass deposition functor reviewed.
- `quokka::DepositionCount::operator()(...)` (`reviewed`): nearest-eight count deposition functor reviewed.
- `SNFeedbackUtils::depositThermalSNR(...)` (`reviewed`): thermal-only deposition kernel reviewed; no additional confirmed bug in inspected implementation.
- `SNFeedbackUtils::depositThermalKineticMomentumSNR(...)` (`finding`, robustness): when `SN_smooth_gas_velocity == false`, the cross-term uses `((px*p_radial_x)+(py*p_radial_y)+(pz*p_radial_z))/rho` (`src/particles/particle_deposition.hpp:301-304`) without guarding `rho <= 0`. In low-density/vacuum cells this can produce `inf`/`nan` energy deposition.
- `SNFeedbackUtils::depositToBuffer(...)` (`reviewed`): particle selection, SN staging, local-buffer deposition, and death-property capture flow reviewed (including valid `AMREX_D_TERM` cell-volume factor macro style at `src/particles/particle_deposition.hpp:389-390`).
- `SNFeedbackUtils::addCompositeBufferToState(...)` (`finding`, robustness): computes `d_e_int_d_rho = e_int / rho` (`src/particles/particle_deposition.hpp:511`) without guarding `rho <= 0`. If a feedback-affected cell has zero/non-positive gas density, this generates invalid energy updates.
- `SNFeedbackUtils::addThermalOnlyBufferToState(...)` (`reviewed`): thermal-only state application path reviewed; no additional confirmed bug in inspected implementation.
- `SNFeedbackUtils::addBufferToState(...)` (`reviewed`): buffer-application dispatch and optional face-centered state handoff reviewed.
- `SNFeedbackUtils::updateEvolutionStageAndDeathDensity(...)` (`reviewed`): stage update + death-density capture path reviewed.
- `SNFeedbackUtils::updateEvolutionStage(...)` (`reviewed`): stage-only update path reviewed.
- `SNDeposition(...)` (`reviewed`): buffer allocation, boundary sum, roundoff, application, and global reduction orchestration reviewed.

### `src/particles/PhysicsParticles.hpp`
- `PhysicsParticleDescriptorBase` (`reviewed`): type-erased particle-descriptor interface, property getters/setters, and default optional virtual methods reviewed.
- `PhysicsParticleDescriptor<ContainerType,problem_t,particleType>` (`reviewed`): constructor and type-tag helper reviewed.
- `PhysicsParticleDescriptor<...>::getParticleDataAtAllLevels()` / `getParticleDataAtLevel(...)` / `getNumParticles()` (`reviewed`): particle I/O/introspection wrappers reviewed.
- `PhysicsParticleDescriptor<...>::computeStellarMass()` / `computeStellarMassAtBirth()` / `computeStellarMassAtBirthBornByTime(...)` (`reviewed`): reduction-based stellar-mass queries reviewed.
- `PhysicsParticleDescriptor<...>::depositMass(...)` / `driftParticles(...)` / `kickParticles(...)` / `destroyParticles(...)` (`reviewed`): 3D mass deposition and particle update/destruction paths reviewed.
- `PhysicsParticleDescriptor<...>::splitParticles(...)` (`finding`, robustness): no validation that `splitFactor > 0` (`src/particles/PhysicsParticles.hpp:408-469`). `splitFactor == 0` marks old particles for deletion and creates none; negative values can also overflow `max_new_particles` (`:414`) and corrupt ID/resize logic.
- `PhysicsParticleDescriptor<...>::computeMaxParticleSpeed(...)` (`reviewed`): particle-speed reduction and cross-rank max path reviewed.
- `PhysicsParticleDescriptor<...>::depositRadiation(...)` / `redistribute(...)` overloads / `writePlotFile(...)` / `writeCheckpoint(...)` / `writeUnitsFile(...)` / `printParticleStatistics()` / `saveParticleDataToTxtFile(...)` (`reviewed`): descriptor wrapper methods reviewed.
- `PhysicsParticleDescriptor<...>::tagCellsAroundParticles(...)` / `updateParticleProperties(...)` / `depositSN(...)` / `computeSinkAccretion(...)` / `applySinkAccretion(...)` / `createParticlesFromState(...)` (`reviewed`): 3D tagging/SN/sink/creation delegation paths reviewed.
- `PhysicsParticleRegister<problem_t>` (`reviewed`): registry storage, registration, dispatch, redistribution, I/O, particle-update orchestration, and statistics methods reviewed.
- `PhysicsParticleRegister::depositRadiation(...)` / `depositMass(...)` / `depositSN(...)` (`reviewed`): per-descriptor aggregation/dispatch paths reviewed.
- `PhysicsParticleRegister::readSFH(...)` (`finding`, robustness): returns `last_time` by overwriting with each parsed entry (`src/particles/PhysicsParticles.hpp:1136-1175`, assignment at `:1165`) rather than tracking the maximum time across particle types/histories. With multiple formation particle types, the returned restart time can depend on iteration order.
- `PhysicsParticleRegister::computePhotoelectricHeatingRate(...)` (`reviewed`): SFH-based PE heating accumulation over registered histories reviewed.

### `src/turbulence/TurbDataReader.hpp`
- `turb_data` (`reviewed`): turbulence field table container (`dvx/dvy/dvz`) reviewed.
- `initialize_turbdata(...)` / `read_dataset(...)` / `get_tabledata(...)` / `computeRms(...)` declarations (`reviewed`): interface signatures reviewed.

### `src/turbulence/TurbDataReader.cpp`
- `read_dataset(...)` (`finding`, resource leak): opens an HDF5 dataspace with `H5Dget_space(...)` (`src/turbulence/TurbDataReader.cpp:26`) but never calls `H5Sclose(dspace)`, leaking an HDF5 handle per dataset read.
- `read_dataset(...)` (`finding`, robustness): reads `ndims` but unconditionally indexes `dims[0..2]` when constructing the 3D table (`src/turbulence/TurbDataReader.cpp:27-30`, `:49`). Malformed/non-3D datasets can trigger out-of-bounds access.
- `initialize_turbdata(...)` (`reviewed`): file open/close and `pertx/perty/pertz` dataset loading flow reviewed.
- `get_tabledata(...)` (`reviewed`): Table3D-to-TableData pinned-memory copy path reviewed.
- `computeRms(...)` (`finding`, robustness): divides by `N` without guarding `N == 0` (`src/turbulence/TurbDataReader.cpp:117`). Empty turbulence tables yield invalid RMS (`nan`/`inf`).

### `src/turbulence/TurbulentDriving.hpp`
- `quokka::turbulence::turbulentDriving<problem_t>` (`reviewed`): driving wrapper state (`TurbGenEx`, `updated`, cached dispersion) and constructor reviewed.
- `quokka::turbulence::turbulentDriving<problem_t>::update(...)` (`reviewed`): driving-update check and dispersion precompute path reviewed.
- `quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...)` (`finding`, correctness/API): computes and stores `updated` (`src/turbulence/TurbulentDriving.hpp:58`) but always returns `true` (`:98`). The return value does not reflect whether the driving field was actually updated/applied.
- `quokka::turbulence::turbulentDriving<problem_t>::applyDriving(...)` (`finding`, robustness): kernel computes velocity and energy increments using division by cell density (`src/turbulence/TurbulentDriving.hpp:81`, `:86`, `:90`) without guarding `rho <= 0`.
- `quokka::turbulence::calculate_dispersion<problem_t>(...)` (`finding`, robustness): particle-free/zero-mass states are not guarded; reductions divide by `total_rho` (`src/turbulence/TurbulentDriving.hpp:143-150`) and can produce `nan`/`inf`.

### `src/turbulence/TurbulentDriving.cpp`
- No function definitions (`reviewed`). File only includes `TurbulentDriving.hpp`.

### `src/QuokkaSimulation.cpp`
- No function definitions (`reviewed`). File only includes `QuokkaSimulation.hpp`.

### `src/particles/stellarpop_data.hpp`
- Constants/type aliases (`Real`, `FATE_ARR_SIZE`, `AGE_ARR_SIZE`, `YR_TO_SEC`) (`reviewed`): static table metadata reviewed.
- `interpolate_whether_SN_explosion(Real)` (`reviewed`): table interpolation path and clamp-policy behavior reviewed; no additional confirmed bug in interpolation logic.
- `interpolate_death_time(Real)` (`reviewed`): stellar lifetime interpolation path and CGS conversion reviewed; no additional confirmed bug in interpolation logic.
- Header structure (`finding`): include guard lines are commented out (`src/particles/stellarpop_data.hpp:7-8`) and there is no `#pragma once`, so accidental multiple inclusion can cause redefinition errors in a translation unit.

### `src/particles/particle_IO.hpp`
- `particle_io::getParticleDataAtAllLevels<ContainerType>(...)` (`reviewed`): all-level particle gather to rank 0 via temporary single-box container and GPU->CPU copy reviewed.
- `particle_io::getParticleDataAtLevel<ContainerType>(...)` (`reviewed`): single-level particle gather path reviewed.
- `particle_io::writeUnitsFile<ContainerType,problem_t,particleType>(...)` (`reviewed`): per-particle-type YAML unit metadata writer reviewed.
- `particle_io::printParticleStatistics<ContainerType,problem_t,particleType>(...)` (`reviewed`): summary + capped per-particle print path reviewed.
- `particle_io::saveParticleDataToTxtFile<ContainerType>(...)` (`finding`, correctness/output): gathers `particle_ids` (`src/particles/particle_IO.hpp:376`) but never writes them, and integer-component output starts at index `1` (`src/particles/particle_IO.hpp:398`) instead of `0`. This silently drops the first user integer component (and all integer data when `NInt == 1`).

### `src/particles/particle_accretion.hpp`
- `AccretionScheme` / `accretion_scheme` (`reviewed`): sink accretion mode enum and compile-time selection reviewed.
- `SinkAccretionUtils` constants (`stencil_size`, `rho_infty_stencil_size`, `r_acc_tolerance`) (`reviewed`): accretion kernel geometry parameters reviewed.
- `SinkAccretionUtils::get_delta_rho(...)` (`reviewed`): helper formula reviewed (currently appears unused in inspected file).
- `SinkAccretionUtils::compute_Mdot_and_r_K<problem_t>(...)` (`finding`, robustness): computes `rho_infty = sum_rho / n_cells`, `vx_grid = sum_px / sum_rho`, and `cs_infty = sum_cs / sum_rho` (`src/particles/particle_accretion.hpp:95-103`) before guarding zero counts/mass. Pathological empty/zero-density stencils can produce `inf`/`nan`.
- `SinkAccretionUtils::compute_accretion_kernel(...)` (`reviewed`): Gaussian kernel helper reviewed.
- `SinkAccretionUtils::ComputeAccretionRateInBox<...>(...)` (`finding`, robustness): relative accretion rate uses `... / (vol * rho)` (`src/particles/particle_accretion.hpp:235-239`) without a runtime guard on `rho > 0` beyond an `AMREX_ASSERT`, so release builds can emit invalid rates if zero-density cells appear.
- `SinkAccretionUtils::ComputeScaleDown<problem_t>(...)` (`finding`, correctness): Jeans-density limiter branch is guarded by `if (accretion_rate_cell > std::numeric_limits<double>::min())` (`src/particles/particle_accretion.hpp:289`), but accretion-zone rates are accumulated as non-positive values (`src/particles/particle_accretion.hpp:237-239`). The intended limiter branch is effectively skipped.
- `SinkAccretionUtils::UpdateParticleMassAndMomentumInBox<...>(...)` (`finding`, robustness): accumulates cell velocities via `mom/rho` (`src/particles/particle_accretion.hpp:415-418`) without a non-assert guard for `rho <= 0`.
- `SinkAccretionUtils::UpdateParticleMassAndMomentum<...>(...)` (`reviewed`): per-tile orchestration and geometry capture reviewed.
- `SinkAccretionUtils::UpdateHydroState<problem_t>(...)` (`reviewed`): conservative hydro-state scaling by accretion factor reviewed.
- `SinkAccretionUtils::computeAccretion<...>(...)` (`reviewed`): per-tile accretion-rate accumulation + `SumBoundary` orchestration reviewed.
- `SinkAccretionUtils::applyAccretion<...>(...)` (`reviewed`): scale-down, particle update, then hydro update sequencing reviewed.

### `src/particles/particle_creation.hpp`
- `ParticleCreationImpl::createParticlesImpl<...>(...)` (`reviewed`): generic checker/count/prefix-sum/create flow reviewed, including `NextID` reservation and per-box particle tile growth.
- `ParticleCreationTraits<particleType>` (default) (`reviewed`): default no-op `ParticleChecker`, `ParticleCreator`, and dispatch wrapper reviewed.
- `ParticleCreationTraits<ParticleType::Sink>::ParticleChecker<problem_t>::operator()(...)` (`reviewed`): Jeans-threshold + local-maximum sink formation criteria reviewed.
- `ParticleCreationTraits<ParticleType::Sink>::ParticleCreator<problem_t>::operator()(...)` (`reviewed`): sink particle initialization and same-cell hydro mass removal path reviewed.
- `ParticleCreationTraits<ParticleType::Sink>::createParticles<...>(...)` (`reviewed`): sink specialization dispatch wrapper reviewed.
- `ParticleCreationTraits<ParticleType::StochasticStellarPop>` constants/config (`reviewed`): IMF/star-formation parameters reviewed.
- `ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleChecker<problem_t>::operator()(...)` (`reviewed`): stochastic star-formation probability and star-count draw logic reviewed.
- `ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleCreator<problem_t>::operator()(...)` (`finding`, robustness/correctness): computes `num_low = ceil(mass_low_mass_star / low_mass_composite_max_mass_)` unconditionally (`src/particles/particle_creation.hpp:499`), but the checker explicitly treats `low_mass_composite_max_mass_ <= 0` as “no splitting” (`src/particles/particle_creation.hpp:444-446`). Non-positive `low_mass_composite_max_mass` can therefore trigger division by zero/invalid `num_low` in the creator path.
- `ParticleCreationTraits<ParticleType::StochasticStellarPop>::ParticleCreator<problem_t>::operator()(...)` (`reviewed`/`partial`): low/high-mass star initialization, IMF sampling, lifetime/fate assignment, low-mass COM correction, and hydro mass-removal scaling reviewed.
- `ParticleCreationTraits<ParticleType::StochasticStellarPop>::createParticles<...>(...)` (`reviewed`): CGS-unit guard and specialization dispatch wrapper reviewed.

### `src/math/interpolate.hpp`
- `binary_search_with_guess(...)` (`reviewed`): search logic and edge-case return conventions are consistent with documented behavior.
- `interpolate_arrays(...)` (`reviewed`): vector interpolation path reviewed; no confirmed bug in inspected implementation.
- `interpolate_value<BoundaryPolicy>(...)` (`reviewed`): boundary-policy handling and interpolation logic reviewed; no confirmed bug in inspected implementation.

### `src/math/FastMath.hpp`
- `FastMath::fastlg(...)` (`reviewed`): approximate log2 helper with positive-input assert.
- `FastMath::fastpow2(...)` (`reviewed`): approximate power-of-two helper.
- `FastMath::lg(...)` (`reviewed`): wrapper over `fastlg`.
- `FastMath::pow2(...)` (`reviewed`): wrapper over `fastpow2`.
- `FastMath::log10(...)` (`reviewed`): approximate base-10 log via `lg`.
- `FastMath::pow10(...)` (`reviewed`): approximate base-10 exponent via `pow2`.
- `FastMath::inverse_pow2(...)` (`reviewed`): Newton iteration helper; no confirmed bug in inspected implementation.

### `src/math/math_impl.hpp`
- `clamp(double,double,double)` (`reviewed`): straightforward helper.
- `sgn(T)` (`reviewed`): straightforward helper.

### `src/linear_advection/linear_advection.cpp`
- No function definitions (`reviewed`). File only includes `linear_advection.hpp`.

### `src/linear_advection/linear_advection.hpp`
- `LinearAdvectionSystem::ComputeMaxSignalSpeed(...)` (`reviewed`): constant-speed norm assignment looks correct for linear advection.
- `LinearAdvectionSystem::ConservedToPrimitive(...)` (`reviewed`): direct copy is expected for scalar advection.
- `LinearAdvectionSystem::isStateValid(...)` (`reviewed`): positivity check on advected density/scalar.
- `LinearAdvectionSystem::PredictStep(...)` (`reviewed`): explicit flux-divergence update follows documented sign convention.
- `LinearAdvectionSystem::AddFluxesRK2(...)` (`reviewed`): SSPRK2 combine step reviewed.
- `LinearAdvectionSystem::ComputeFluxes<DIR>(...)` (`reviewed`): upwind flux and face velocity logic appears internally consistent (and was used as a reference to flag hydro face-velocity bug).

### `src/linear_advection/AdvectionSimulation.cpp`
- No function definitions (`reviewed`). File only includes `AdvectionSimulation.hpp`.

### `src/linear_advection/AdvectionSimulation.hpp`
- `AdvectionSimulation::computeMaxSignalLocal(int)` (`finding`): uses `state_old_cc_[level]` (`src/linear_advection/AdvectionSimulation.hpp:134`) instead of the current state (`state_new_cc_`) when computing CFL speeds, unlike `QuokkaSimulation`. This can make the timestep estimate use stale data.
- `AdvectionSimulation::printCellProperties(...)` (`reviewed`): deliberate no-op.
- `AdvectionSimulation::fillPoissonRhsAtLevel(...)` (`reviewed`): deliberate no-op.
- `AdvectionSimulation::applyPoissonGravityAtLevel(...)` (`reviewed`): deliberate no-op.
- Hook/default methods (`preCalculateInitialConditions`, `setInitialConditionsOnGrid`, `setInitialConditionsOnGridFaceVars`, `createInitial*Particles`, `computeBeforeTimestep`, `computeAfterTimestep`, `ComputeDerivedVar`, `ComputeStatistics`, `refineGrid`, `ErrorEst`, `FixupState`, `computeReferenceSolution`) (`reviewed`): default or thin-wrapper implementations reviewed; no confirmed bug in inspected sections.
- `AdvectionSimulation::computeAfterEvolve(...)` (`finding`, robustness): computes `rel_error = err_norm / sol_norm` without guarding `sol_norm == 0` (`src/linear_advection/AdvectionSimulation.hpp:283`). Degenerate zero-reference solutions can produce `inf`/`nan` error norms.
- `AdvectionSimulation::advanceSingleTimestepAtLevel(...)` (`reviewed`): RK2 flow, ghost fills, and reflux accumulation reviewed; no additional confirmed bug in inspected section.
- `AdvectionSimulation::computeFluxes(...)` (`reviewed`): temporary allocation and directional dispatch look consistent.
- `AdvectionSimulation::fluxFunction<DIR>(...)` (`reviewed`): reconstruct + flux dispatch logic reviewed.
- `AdvectionSimulation::WriteSingleLevelPlotfileSimplified(...)` (`reviewed`): wrapper logic is straightforward.

### `src/io/io_utils.hpp`
- `quokka::ScopedVisMFNOutFiles` ctor/dtor (`reviewed`): RAII save/restore of `VisMF::NOutFiles` looks correct.

### `src/io/DiagFilter.cpp`
- `DiagFilter::init(...)` (`reviewed`): filter field/range parsing logic reviewed; aborts on missing config.
- `DiagFilter::setup(...)` (`reviewed`): field-name to index resolution is straightforward.

### `src/io/DiagBase.cpp`
- `DiagBase::init(...)` (`reviewed`): interval parsing and filter initialization reviewed; time-based diagnostics start at one interval (consistent with plotfile behavior).
- `DiagBase::prepare(...)` (`finding`): gated by `if (first_time)` (`src/io/DiagBase.cpp:38`) but never sets `first_time = false`. As a result, the one-time filter-device setup path re-runs on every `prepare()` call (and base-class users like `DiagPlotfile`/`DiagParticleTxt` rely on this method directly).
- `DiagBase::doDiag(...)` (`reviewed`): step/time scheduling logic reviewed; no additional confirmed bug in inspected path.
- `DiagBase::addVars(...)` (`reviewed`): straightforward filter-var append.
- `DiagBase::getFieldIndex(...)` (`reviewed`): straightforward lookup with abort on missing field.
- `DiagBase::getFieldIndexVec(...)` (`reviewed`): straightforward helper wrapper.

### `src/io/DiagPDF.H`
- `DiagPDF::identifier()` (`reviewed`): static registration identifier.
- `DiagPDF::getBinIndex1D(...)` (`reviewed`/`partial`): bin-index helper is straightforward for valid transformed inputs.
- `DiagPDF::getTotalBinCount()` (`reviewed`): multiplies per-dimension bin counts.
- `DiagPDF::processDiag<problem_t>(...)` (`reviewed`): histogram setup, masking, and accumulation path reviewed; no additional confirmed bug beyond the log-bin robustness note below.

### `src/io/DiagPDF.cpp`
- `DiagPDF::init(...)` (`reviewed`): histogram parameter parsing reviewed; explicit-range positivity checks are present for log bins.
- `DiagPDF::addVars(...)` (`reviewed`): adds `gasDensity` for mass-weighted histograms and requested fields.
- `DiagPDF::prepare(...)` (`reviewed`): caches geometry/refinement metadata; correctly clears `first_time`.
- `DiagPDF::getIdxVec(...)` (`reviewed`): inverse linear-index mapping looks correct.
- `DiagPDF::MFVecMin(...)` (`reviewed`): level-wise min reduction helper.
- `DiagPDF::MFVecMax(...)` (`reviewed`): level-wise max reduction helper.
- `DiagPDF::writePDFToFile(...)` (`reviewed`): file formatting and bin-edge reconstruction reviewed; no confirmed bug in inspected implementation.
- `DiagPDF` log-bin path (`finding`, robustness): when `log_spaced_bins=1` with auto-ranged bounds (`m_useFieldMinMax[n] == true`), `processDiag()` computes `log10(m_lowBnd[n])` / `log10(m_highBnd[n])` without validating positivity (`src/io/DiagPDF.H` template path around transformed range setup). Non-positive field extrema can produce `nan`/`-inf` and corrupt histogram indexing.

### `src/io/DiagParticleTxt.cpp`
- `DiagParticleTxt::init(...)` (`reviewed`): parameter parsing and file-prefix normalization reviewed.
- `DiagParticleTxt::prepare(...)` (`reviewed`): thin wrapper around `DiagBase::prepare()` for filter setup only.
- `DiagParticleTxt::addVars(...)` (`reviewed`): intentional no-op for standard field extraction.

### `src/io/DiagParticleTxt.H`
- `DiagParticleTxt::identifier()` (`reviewed`): static registration identifier.
- `DiagParticleTxt::close()` (`reviewed`): no-op.
- `DiagParticleTxt::processDiag<problem_t>(...)` (`finding`): header comment/member comment say empty `m_particleTypes` means "all" (`src/io/DiagParticleTxt.H:30`), and `init()` prints "Including all particle types" when none are specified; but `processDiag()` skips output when the list is empty (`src/io/DiagParticleTxt.H:55-60`). Default configuration therefore emits no particle diagnostic.

### `src/io/DiagPlotfile.cpp`
- `DiagPlotfile::init(...)` (`reviewed`): runtime parsing for file prefix, particle selection, field selection, and fc-field toggle reviewed.
- `DiagPlotfile::prepare(...)` (`reviewed`): thin wrapper around `DiagBase::prepare()` for filter setup only.
- `DiagPlotfile::addVars(...)` (`reviewed`): intentional no-op because this diagnostic writes full plotfile data directly.

### `src/io/DiagPlotfile.H`
- `DiagPlotfile::identifier()` (`reviewed`): static registration identifier.
- `DiagPlotfile::close()` (`reviewed`): no-op.
- `DiagPlotfile::getDiagFileName()` / `getParticleTypes()` (`reviewed`): simple accessors.
- `DiagPlotfile::processDiag<problem_t>(...)` (`finding`, conditional): when `QUOKKA_USE_OPENPMD` is enabled and `field_names` filtering is requested, the OpenPMD path writes the unfiltered `varnames` + `mf_cc_ptr` (`src/io/DiagPlotfile.H:127`) instead of `varnames_out` + `mf_cc_out_ptr` used by the AMReX plotfile path (`src/io/DiagPlotfile.H:136`). `field_names` is silently ignored for OpenPMD output.
- `DiagPlotfile::processDiag<problem_t>(...)` (`reviewed`): reviewed CC filtering, metadata write, fc_vars output, and particle output paths; no additional confirmed bug in inspected implementation.
- `DiagPlotfile::WriteMetadataFile(...)` (`reviewed`): straightforward YAML metadata writer.

### `src/io/DiagProjectionPlot.cpp`
- `DiagProjectionPlot::init(...)` (`reviewed`): parameter parsing, field validation setup, filter discard behavior, and projection-direction parsing reviewed.
- `DiagProjectionPlot::prepare(...)` (`reviewed`): field existence validation and base prepare wrapper reviewed; no standalone confirmed bug beyond repeated base `first_time` issue already logged for `DiagBase`.
- `DiagProjectionPlot::addVars(...)` (`reviewed`): appends requested projection fields.

### `src/io/DiagProjectionPlot.H`
- `DiagProjectionPlot::identifier()` (`reviewed`): static registration identifier.
- `DiagProjectionPlot::close()` (`reviewed`): no-op.
- `DiagProjectionPlot::getParticleTypes()` (`reviewed`): simple accessor.
- `DiagProjectionPlot::processDiag<problem_t>(...)` (`reviewed`): reviewed projection assembly, plot writing, and optional particle write path; no confirmed bug in inspected implementation.

### `src/io/DiagFramePlane.cpp`
- `printLowerDimIntVect(...)` (`reviewed`): helper for 2D header formatting.
- `printLowerDimBox(...)` (`reviewed`): helper for 2D header formatting.
- `DiagFramePlane::init(...)` (`finding`, low): filter-warning condition is reversed. It prints "filters ... will be discarded" only when `m_filters.empty()` (`src/io/DiagFramePlane.cpp:47`), and does not clear filters when they are actually present. Users specifying filters get no warning despite filters being unsupported here.
- `DiagFramePlane::addVars(...)` (`reviewed`): appends requested slice fields.
- `DiagFramePlane::prepare(...)` (`reviewed`): reviewed field validation, plane geometry setup, interpolation weights, and slice BA/DM construction; no additional confirmed bug in inspected implementation.
- `DiagFramePlane::Write2DMultiLevelPlotfile(...)` (`reviewed`): custom 2D plotfile writer workflow reviewed.
- `DiagFramePlane::Write2DPlotfileHeader(...)` (`reviewed`): 2D header serialization reviewed.
- `DiagFramePlane::VisMF2D(...)` (`reviewed`): 2D MultiFab writer implementation reviewed.
- `DiagFramePlane::Write2DMFHeader(...)` (`reviewed`): helper header writer reviewed.
- `DiagFramePlane::Find2FOffsets(...)` (`reviewed`): offset bookkeeping mirrors AMReX-style flow.
- `DiagFramePlane::write_2D_header(...)` (`reviewed`): helper routine reviewed.

### `src/io/DiagFramePlane.H`
- `DiagFramePlane::identifier()` (`reviewed`): static registration identifier.
- `DiagFramePlane::getParticleTypes()` (`reviewed`): simple accessor.
- `DiagFramePlane::close()` (`reviewed`): no-op.
- `DiagFramePlane::processDiag<problem_t>(...)` (`reviewed`): reviewed interpolation and output orchestration; no confirmed bug in inspected implementation.

### `src/io/projection.hpp`
- `quokka::diagnostics::detail::*` declarations (`reviewed`): declarations match corresponding implementation set in `projection.cpp`.
- `quokka::diagnostics::ComputePlaneProjectionFromMultiFab(...)` (`reviewed`): reviewed masking, plane reduction, coarse-to-fine accumulation, and 2D refinement-ratio mapping; no confirmed bug in inspected implementation.
- `quokka::diagnostics::WriteProjection(...)` declaration (`reviewed`): non-template declaration only.
- `quokka::diagnostics::WriteProjection(..., particleRegister, ...)` template overload (`reviewed`): wrapper delegates to base writer, then writes filtered particles when requested.

### `src/io/projection.cpp`
- `quokka::diagnostics::detail::direction_to_string(...)` (`reviewed`): guarded enum->string helper.
- `quokka::diagnostics::detail::printLowerDimIntVect(...)` / `printLowerDimBox(...)` (`reviewed`): formatting helpers.
- `quokka::diagnostics::detail::Write2DMultiLevelPlotfile(...)` (`reviewed`): 2D AMReX plotfile writer flow reviewed.
- `quokka::diagnostics::detail::Write2DPlotfileHeader(...)` (`reviewed`): header serialization reviewed.
- `quokka::diagnostics::detail::VisMF2D(...)` (`reviewed`): writer implementation mirrors `DiagFramePlane` variant; no confirmed bug in inspected implementation.
- `quokka::diagnostics::detail::Write2DMFHeader(...)` (`reviewed`): header helper reviewed.
- `quokka::diagnostics::detail::Find2FOffsets(...)` (`reviewed`): offset bookkeeping reviewed.
- `quokka::diagnostics::detail::write_2D_header(...)` (`reviewed`): helper routine reviewed.
- `quokka::diagnostics::detail::transform_box_to_2D(...)` (`reviewed`): direction-dependent index transform reviewed.
- `quokka::diagnostics::detail::transform_realbox_to_2D(...)` (`reviewed`): direction-dependent domain transform reviewed.
- `quokka::diagnostics::detail::transform_ref_ratio_to_2D(...)` (`reviewed`): direction-dependent ref-ratio transform reviewed.
- `quokka::diagnostics::WriteProjection(...)` (`reviewed`): reviewed variable collation, union-BoxArray fallback, and metadata write path; no confirmed bug in inspected implementation.

### `src/io/openPMD.hpp`
- `quokka::OpenPMDOutput::detail::*` declarations (`reviewed`): helper declarations reviewed.
- `quokka::OpenPMDOutput::WriteFile(...)` declaration (`reviewed`): API declaration reviewed.

### `src/io/openPMD.cpp`
- `quokka::OpenPMDOutput::detail::getReversedVec(IntVect)` (`reviewed`): dimension reversal helper reviewed.
- `quokka::OpenPMDOutput::detail::getReversedVec(Real*)` (`reviewed`): pointer-to-vector reversal helper reviewed.
- `quokka::OpenPMDOutput::detail::SetupMeshComponent(...)` (`reviewed`): mesh metadata setup reviewed; inline comment correctly notes ghost-zone offsets would overflow unsigned chunk offsets if used.
- `quokka::OpenPMDOutput::detail::GetMeshComponentName(...)` (`reviewed`): sanitizes field names and appends level suffixes.
- `quokka::OpenPMDOutput::WriteFile(...)` (`reviewed`): reviewed openPMD series/iteration creation and chunked MultiFab writes; no additional confirmed bug beyond the `DiagPlotfile` OpenPMD field-filter mismatch already logged.

### `src/util/ArrayUtil.hpp`
- `strided_vector_from(...)` (`finding`, robustness): no validation for `stride <= 0` (`src/util/ArrayUtil.hpp:16`). A zero stride causes an infinite loop; a negative stride underflows the unsigned loop index progression.

### `src/util/ArrayView.hpp`
- Include dispatch (`reviewed`): dimensional include selection is straightforward.

### `src/util/ArrayView_2d.hpp`
- `quokka::reorderMultiIndex<FluxDir::X1/X2>(...)` (`reviewed`): permutation helpers match 2D view semantics.
- `quokka::Array4View<...>` specializations (`reviewed`): const/non-const index-permuting wrappers reviewed; no confirmed bug in inspected implementation.

### `src/util/ArrayView_3d.hpp`
- `quokka::reorderMultiIndex<FluxDir::X1/X2/X3>(...)` (`reviewed`): permutation helpers match documented 3D view semantics.
- `quokka::Array4View<...>` specializations (`reviewed`): const/non-const wrappers for X1/X2/X3 views reviewed; no confirmed bug in inspected implementation.

### `src/util/BC.hpp`
- `quokka::detail::isNormalComponent<problem_t>(...)` (`reviewed`): reflecting-boundary component classification logic reviewed.
- `quokka::BC<problem_t>(int,int,int)` / `BC<problem_t>(int)` (`reviewed`): cell-centered BC construction and reflecting special-case logic reviewed.
- `quokka::BC_cc<problem_t>(...)` (`reviewed`): enum overload mirrors integer BC builder.
- `quokka::BC_fc<problem_t>(...)` (`reviewed`): face-centered BC builder reviewed; reflecting MHD fallback-to-even is explicitly marked TODO.

### `src/util/CheckNaN.hpp`
- `quokka::CheckSymmetryArray<T>(...)` (`reviewed`): default stub returns true for problem specializations.
- `quokka::CheckSymmetryFluxes<T>(...)` (`reviewed`): default stub returns true for problem specializations.
- `quokka::CheckNaN<T>(...)` (`reviewed`): GPU `contains_nan` assertion wrapper reviewed.

### `src/util/Optional.hpp`
- `quokka::optional<T>` ctors/assignment/dtor/`operator bool`/`operator*` (`reviewed`): minimal GPU-compatible optional implementation reviewed; no confirmed bug in inspected implementation.

### `src/util/richardson.hpp`
- `quokka::richardson::Parameters` (`reviewed`): parameter POD.
- `quokka::richardson::applyQuietDefaults()` (`reviewed`): ParmParse defaults helper reviewed.
- `quokka::richardson::run(...)` (`reviewed`/`partial`): Richardson driver reviewed; no confirmed bug in inspected implementation this pass.

### `src/util/fextract.hpp`
- `fextract(...)` declaration (`reviewed`): API declaration reviewed.

### `src/util/fextract.cpp`
- `fextract(...)` (`reviewed`): reviewed slice-index selection, contiguous packing, MPI gather/sort path, and GPU copy kernels; no confirmed bug in inspected implementation this pass.

### `src/util/valarray.hpp`
- `quokka::valarray<T,d>` core methods (`reviewed`): initializer-list ctor, indexing, `fillin`, `hasnan` reviewed.
- Arithmetic/comparison helpers (`operator+,-,*,/,+=,*=,/=`, `abs`, `min`, `max`, `sum`, comparisons) (`reviewed`): utility operators reviewed; no confirmed bug in inspected implementation.

### `src/util/DataTable.hpp`
- `quokka::InterpData<Ndim>` (`reviewed`): interpolation metadata carrier.
- `quokka::DataTableGpuConst<Ndim,Nout,oob_policy>::find_interpolation_data(...)` (`finding`, robustness): assumes at least 2 points per dimension by forcing `indices = sizes-2` at the upper edge (`src/util/DataTable.hpp:123-126`) and dividing by `dcoord[dim]` (`src/util/DataTable.hpp:130-131`). If any dimension size is 1, this produces invalid indices/divide-by-zero.
- `quokka::DataTableGpuConst<...>::interpolate(...)` / `interpolate_single(...)` / interpolation helpers (`reviewed`): reviewed spacing transforms and n-linear interpolation logic for 1D-4D paths; no additional confirmed bug in inspected implementation.
- `quokka::DataTable<Ndim,Nout,oob_policy>` constructors / move ops / metadata accessors / `const_tables()` / `is_initialized()` / size accessors (`reviewed`): reviewed core object lifecycle and accessors.
- `quokka::DataTable<...>::initialize_common(...)` (`finding`, robustness): validates only `sizes_[dim] > 0` (`src/util/DataTable.hpp:721`) but later computes `dcoord_[dim] = ... / (sizes_[dim]-1)` (`src/util/DataTable.hpp:763`). `size==1` tables are accepted but create zero-division/invalid interpolation state.
- `quokka::DataTable<...>::CSVReader(...)` (`reviewed`): reviewed metadata parsing and dimensional data-load/transposition logic through 4D paths; no additional confirmed bug in inspected sections beyond existing size-1 table robustness findings.
- `quokka::DataTable<...>::H5Reader(...)` (`reviewed`): reviewed HDF5 metadata/attribute reads, optional `include_pe` parsing, coordinate/dataset loads, dimensional unpacking (1D-4D), and file close paths; no additional confirmed bug in inspected implementation this pass.

### `src/util/matplotlibcpp.h`
- Vendored third-party header (`partial`): targeted audit of Python/NumPy interop and stateful plotting paths only (not an exhaustive line-by-line validation of all wrappers).
- `matplotlibcpp::detail::_interpreter` (`partial`): singleton interpreter init/import wiring reviewed; no new confirmed bug in inspected constructor/import path this pass.
- `matplotlibcpp::detail::_interpreter::safe_import(...)` (`reviewed`): validates imported attribute exists and is a Python function.
- `matplotlibcpp::backend(...)` (`reviewed`): simple backend setter.
- `matplotlibcpp::annotate(...)` (`reviewed`/`partial`): wrapper call pattern reviewed.
- `matplotlibcpp::select_npy_type<...>` specializations (`reviewed`): type mapping traits reviewed.
- `matplotlibcpp::get_array(const std::vector<Numeric>&)` (`finding`): in the NumPy path for unsupported element types (`NPY_NOTYPE`), it builds a local temporary `std::vector<double> vd` and returns `PyArray_SimpleNewFromData(..., vd.data())` (`src/util/matplotlibcpp.h:316-320`). The returned NumPy array then points to freed stack storage after the function returns (dangling pointer / use-after-free).
- `matplotlibcpp::get_2darray(...)` (`reviewed`/`partial`): allocates Python-owned NumPy array and copies rows; shape consistency checks present.
- Plot wrapper family (`plot`, `stem`, `fill`, `fill_between`, `hist`, `scatter`, `bar`, `named_*`, `semilog*`, `loglog`, `text`, variadic `plot`, etc.) (`partial`): spot-checked representative wrappers for argument construction/call patterns; no additional confirmed bug logged from this pass beyond issues below.
- `matplotlibcpp::internal::imshow(...)` / `imshow(...)` overloads (`reviewed`/`partial`): pointer-based wrappers and OpenCV bridge reviewed; no confirmed bug in inspected paths.
- `matplotlibcpp::subplots_adjust(...)` (`reviewed`): simple keyword dispatch wrapper.
- `matplotlibcpp::figure(...)`, `fignum_exists(...)`, `figure_size(...)`, `legend(...)`, `xscale(...)`, `yscale(...)`, `xlim(left,right)`, `ylim(left,right)`, `xticks(...)`, `yticks(...)`, `tick_params(...)`, `title(...)`, `suptitle(...)`, `axis(...)`, `xlabel(...)`, `ylabel(...)`, `grid(...)`, `show(...)`, `close(...)`, `xkcd(...)`, `draw(...)`, `pause(...)`, `save(...)`, `clf(...)`, `ion(...)`, `ginput(...)`, `tight_layout(...)` (`partial`): representative wrappers reviewed; see explicit `xlim()/ylim()` and `subplot2grid()` findings below.
- `matplotlibcpp::xlim()` (`finding`, robustness/leak): dereferences `res` via `PyTuple_GetItem(res, ...)` before checking whether the Python call failed (`src/util/matplotlibcpp.h:1305-1314`), which can segfault on error. It also leaks `args` (never `Py_DECREF(args)`), and returns a raw `new double[2]` requiring caller-managed deletion.
- `matplotlibcpp::ylim()` (`finding`, robustness/leak): same issues as `xlim()` (`src/util/matplotlibcpp.h:1322-1334`): null-check happens after dereference, `args` tuple leaked, raw heap array returned.
- `matplotlibcpp::subplot2grid(...)` (`finding`): `PyTuple_SetItem(args, 0/1, shape/loc)` steals references, but the code then manually decrefs `shape` and `loc` (`src/util/matplotlibcpp.h:1480-1491`), causing refcount underflow / premature free / double-decref risk.
- `matplotlibcpp::detail::{is_function,is_callable_impl,is_callable,plot_impl}` (`partial`): metaprogramming dispatch for variadic `plot` reviewed at a high level; no confirmed bug in inspected implementation this pass.
- `matplotlibcpp::Plot` (`finding`, refcount ownership): constructor stores `line = PyList_GetItem(res, 0)` (`src/util/matplotlibcpp.h:1915`) without `Py_INCREF`, but later `decref()` unconditionally `Py_DECREF(line)` (`src/util/matplotlibcpp.h:1969-1970`). `PyList_GetItem` returns a borrowed reference, so ownership semantics are unsafe and can over-decrement.
- `matplotlibcpp::Plot::update(...)` (`finding`, leak): allocates `plot_args` (`src/util/matplotlibcpp.h:1936`) and never decrefs it before returning (`src/util/matplotlibcpp.h:1940-1943`), leaking a Python tuple on each update call.
- `matplotlibcpp::Plot::remove(...)` (`finding`, leak): obtains `remove_fct` and allocates `args` (`src/util/matplotlibcpp.h:1955-1957`) but never decrefs either object, leaking Python references.
- `matplotlibcpp::Plot::clear()` / `remove()` / destructor / `decref()` (`partial`): reviewed lifecycle wrappers; no additional confirmed bug beyond the refcount issues logged above.

### `src/math/Interpolate2D.cpp`
- No function definitions (`reviewed`). File contains comments only.

### `src/math/Interpolate2D.hpp`
- `interpolate2d(...)` (`finding`): boundary-degenerate branch conditions mistakenly compare coordinate `yi` (double) instead of index `iy` (int) (`src/math/Interpolate2D.hpp:58`, `src/math/Interpolate2D.hpp:63`). This misroutes edge-case interpolation logic near table boundaries.

### `src/math/ODEIntegrate.hpp`
- `rk12_single_step(...)` (`reviewed`): Heun/Euler embedded step implementation reviewed.
- `rk23_single_step(...)` (`reviewed`): Bogacki-Shampine (2)3 step implementation reviewed.
- `error_norm(...)` (`finding`, robustness): weighting uses `reltol * y0[i] + abstol[i]` (`src/math/ODEIntegrate.hpp:118`) instead of `reltol * abs(y0[i]) + abstol[i]` (as in standard weighted RMS norms). Negative state values can shrink/cancel the denominator and distort timestep control.
- `rk_adaptive_integrate(...)` (`finding`, robustness): ignores the return code of the initial `rhs(t0, y0, ydot0)` call used for timestep estimation (`src/math/ODEIntegrate.hpp:137`), so an RHS failure can leave `ydot0` invalid and contaminate `dt_guess` / subsequent integration control.
- `rk_adaptive_integrate(...)` (`reviewed`): adaptive loop and retry controller reviewed; no additional confirmed bug in inspected implementation beyond the initial-RHS error handling issue.

### `src/math/quadrature.hpp`
- `kernel_wendland_c2(...)` (`reviewed`): compact-support kernel helper reviewed.
- `quad_3d(...)`, `quad_2d(...)`, `quad_1d(...)` (`reviewed`): nested Gauss quadrature wrappers reviewed.

### `src/math/gauss.hpp`
- `quokka::math::quadrature::detail::gauss_constant_category<T>` (`reviewed`): constant-category trait reviewed.
- `quokka::math::quadrature::detail::gauss_detail<...>` tables (`partial`): reviewed structure of fixed abscissa/weight specializations and API pattern; not exhaustively checked every numeric table entry.
- `quokka::math::quadrature::gauss<Real,N>::integrate(...)` overloads (`reviewed`/`partial`): reviewed finite/infinite-limit transform logic and symmetric quadrature accumulation; no confirmed bug in inspected implementation this pass.

### `src/math/root_finding.hpp`
- `quokka::math::eps_tolerance<T>` (`reviewed`): tolerance functor constructors and predicate reviewed.
- `quokka::math::detail::{bracket,safe_div,secant_interpolate,quadratic_interpolate,cubic_interpolate}(...)` (`reviewed`): root-bracketing/interpolation helpers reviewed; no confirmed bug in inspected implementation this pass.
- `quokka::math::toms748_solve(F,ax,bx,fax,fbx,tol,max_iter)` (`reviewed`): reviewed full control flow (secant/quadratic/cubic steps, fallback bisection, iteration bookkeeping, bracketing updates); no confirmed bug in inspected implementation this pass.
- `quokka::math::toms748_solve(F,ax,bx,tol,max_iter)` (`reviewed`): thin overload delegates after precomputing endpoint function values.

### `src/cooling/PhotoelectricHeating.hpp`
- `quokka::detail::seconds_per_year_local` (`reviewed`): local constant used to avoid circular include.
- `quokka::PeHeatingGpuConstTables<...>` (`reviewed`): GPU table bundle POD.
- `quokka::PeHeatingTables<...>::const_tables()` (`reviewed`): wrapper returning GPU-const tables.
- `quokka::PeHeatingTables<...>::is_initialized()` (`reviewed`): simple pass-through.
- `quokka::g_pe_heating_tables_ptr<...>` (`reviewed`): global pointer declaration for active PE tables.
- `quokka::PeHeatingFromSfh(...)` (`finding`, robustness): divides by `sf_area_kpc2` without validating it is positive/non-zero (`src/cooling/PhotoelectricHeating.hpp:77`). A zero area configuration will produce `inf`/`nan` heating rates.
- `quokka::PeHeatingFromConstSfr(...)` (`reviewed`/`partial`): constant-SFR accumulation logic reviewed; no confirmed runtime bug in inspected implementation (comment says 100 Myr while code integrates to 1 Gyr).

### `src/cooling/ResampledCooling.hpp`
- `quokka::ResampledCooling::resampledGpuConstTables` (`reviewed`): GPU table bundle POD.
- `quokka::ResampledCooling::resampled_tables` (`reviewed`/`partial`): host-side table storage and metadata carrier.
- `quokka::ResampledCooling::resampled_cooling_function(...)` (`finding`, robustness): computes `eint = Eint / rho` and `fastlg(rho/eint)` without guarding `rho > 0` and `eint > 0` (`src/cooling/ResampledCooling.hpp:69-70`). Invalid states can produce divide-by-zero/NaNs and invalid table lookups.
- `quokka::ResampledCooling::ComputeTgasFromEgas(...)` (`reviewed`/`partial`): same domain assumptions as cooling function; no additional confirmed bug beyond shared robustness issue.
- `quokka::ResampledCooling::ComputeCoolingLength(...)` (`reviewed`/`partial`): cooling-length helper reviewed; shares unguarded `rho/eint` log-domain assumptions.
- `quokka::ResampledCooling::ComputePressureFromRhoEint(...)` (`reviewed`/`partial`): helper reviewed; shares unguarded `rho/eint` log-domain assumptions.
- `quokka::ResampledCooling::ComputeEntropyFromRhoEint(...)` (`reviewed`/`partial`): helper reviewed; shares unguarded `rho/eint` log-domain assumptions.
- `quokka::ResampledCooling::ComputeSoundSpeedFromRhoEint(...)` (`reviewed`/`partial`): helper reviewed; shares unguarded `rho/eint` log-domain assumptions.
- `quokka::ResampledCooling::ResampledCoolingFunctor` ctors/assignments/dtor (`reviewed`): POD-like functor lifecycle boilerplate.
- `quokka::ResampledCooling::ResampledCoolingFunctor::operator()(...)` (`reviewed`/`partial`): RHS wrapper delegates to table cooling function and constant heating.
- `quokka::ResampledCooling::computeCooling<problem_t>(...)` (`reviewed`): reviewed RK adaptive integration loop, substep tracking, energy update, and retry signaling; no additional confirmed bug beyond inherited low-state robustness assumptions.
- `quokka::ResampledCooling::readResampledData(...)` declaration (`reviewed`): declaration only.

### `src/cooling/ResampledCooling.cpp`
- `quokka::ResampledCooling::cloudy_H_mass_fraction` (`reviewed`): compile-time constant.
- `quokka::ResampledCooling::readResampledData(...)` (`reviewed`): HDF5 table loading/metadata logging path reviewed; no confirmed bug in inspected implementation this pass.
- `quokka::ResampledCooling::resampled_tables::const_tables()` (`reviewed`): aggregates host tables + metadata into GPU-const bundle.

### `src/chemistry/Chemistry.cpp`
- `quokka::chemistry::chemburner(...)` (`reviewed`): thin wrapper around Microphysics `burner(...)`.

### `src/chemistry/Chemistry.hpp`
- `quokka::chemistry::chemburner(...)` declaration (`reviewed`): declaration only.
- `quokka::chemistry::computeChemistry<problem_t>(...)` (`finding`, robustness): the kernel divides species partial densities by `rho` (`src/chemistry/Chemistry.hpp:59-62`) and computes derived fractions (`src/chemistry/Chemistry.hpp:70-72`) before the low-density early-return guard (`src/chemistry/Chemistry.hpp:75-78`). If `rho <= 0` (or extremely small), this can generate invalid values before the intended density cutoff check.
- `quokka::chemistry::computeChemistry<problem_t>(...)` (`reviewed`): reviewed burn invocation, failure reduction, positivity/normalization passes, charge-conservation electron update, EOS recomputation, and state writeback; no additional confirmed bug in inspected implementation this pass.

### `src/problems/Cooling/testCooling.cpp`
- `CoolingTest` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<CoolingTest>` / `Physics_Traits<CoolingTest>` / `SimulationData<CoolingTest>` specializations (`reviewed`): trait and user-data declarations reviewed.
- `QuokkaSimulation<CoolingTest>::preCalculateInitialConditions()` (`reviewed`/`partial`): random phase table allocation/population and host->device copy reviewed; no confirmed bug in inspected implementation.
- `QuokkaSimulation<CoolingTest>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): perturbation IC generation and conservative state initialization reviewed; no confirmed bug in inspected implementation.
- `AMRSimulation<CoolingTest>::setCustomBoundaryConditions(...)` (`reviewed`): custom upper-boundary Dirichlet fill helper usage reviewed.
- `problem_main()` (`reviewed`): manual mixed-BC setup and simulation run orchestration reviewed; no confirmed bug in inspected implementation.

### `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp`
- `ResampledCoolingTest` (`reviewed`): empty tag type for trait specialization.
- `readReferenceCSV(...)` (`reviewed`/`partial`): CSV parsing with header skip and malformed-line tolerance reviewed.
- `SimulationData<ResampledCoolingTest>` / `quokka::EOS_Traits<ResampledCoolingTest>` / `Physics_Traits<ResampledCoolingTest>` specializations (`reviewed`): trait and user-data declarations reviewed.
- `QuokkaSimulation<ResampledCoolingTest>::setInitialConditionsOnGrid(...)` (`reviewed`): isochoric cooling IC setup reviewed.
- `QuokkaSimulation<ResampledCoolingTest>::computeAfterTimestep()` (`reviewed`/`partial`): center extraction and table-based temperature sampling reviewed; no confirmed bug in inspected implementation.
- `problem_main()` (`finding`, robustness): reference comparison computes `rel_error = err_norm / sol_norm` without guarding `sol_norm == 0` (`src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp:221-228`). A degenerate/zero reference dataset would produce `inf`/`nan`.
- `problem_main()` (`reviewed`): runtime parameter parsing, evolve/analyze path, CSV output, and optional plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/PrimordialChem/testPrimordialChem.cpp`
- `PrimordialChemTest` (`reviewed`): empty tag type for trait specialization.
- `Physics_Traits<PrimordialChemTest>` / `SimulationData<PrimordialChemTest>` specializations (`reviewed`): trait and user-data declarations reviewed.
- `QuokkaSimulation<PrimordialChemTest>::preCalculateInitialConditions()` (`reviewed`): microphysics/EOS/network init and ParmParse species setup reviewed.
- `QuokkaSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(...)` (`finding`, robustness): if configured species number densities sum to zero, `rhotot` remains zero and normalization divides by `rhotot` (`src/problems/PrimordialChem/testPrimordialChem.cpp:197-211`), producing invalid initial mass fractions/number densities before EOS call.
- `QuokkaSimulation<PrimordialChemTest>::setInitialConditionsOnGrid(...)` (`reviewed`): species switch initialization, EOS setup, and grid state writeback reviewed; no additional confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simulation setup/evolve orchestration reviewed; no confirmed bug in inspected implementation.

### `src/problems/ODEIntegration/testODEIntegration.cpp`
- `ODETest` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ODETest>` specialization (`reviewed`): trait constants reviewed.
- `ODEUserData` (`reviewed`): unused/simple POD declaration.
- `cooling_function(...)` (`reviewed`): Koyama-Inutsuka cooling/heating fit helper reviewed.
- `ODECoolingFunctor` ctor and `operator()(...)` (`reviewed`): temperature->cooling RHS wrapper reviewed.
- `problem_main()` (`reviewed`/`partial`): EOS init, adaptive ODE solve, and equilibrium-temperature check reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/HydroShocktube/testHydroShocktube.cpp`
- `ShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ShocktubeProblem>` / `Physics_Traits<ShocktubeProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): left/right Riemann IC setup reviewed.
- `AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant Dirichlet left/right boundary state setup reviewed.
- `QuokkaSimulation<ShocktubeProblem>::refineGrid(...)` (`reviewed`/`partial`): density-gradient tagging criterion reviewed.
- `QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): external exact-solution read, interpolation, reference fill, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simulation run and error-tolerance check reviewed.

### `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp`
- `ShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- Global `consv_test_passes` (`reviewed`): global status flag for scalar-conservation test result.
- `SimulationData<ShocktubeProblem>` / `quokka::EOS_Traits<ShocktubeProblem>` / `Physics_Traits<ShocktubeProblem>` specializations (`reviewed`): trait and user-data declarations reviewed.
- `QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): CMA shocktube IC with three species partial densities reviewed.
- `AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)` (`finding`): species-2 partial density at both boundaries is computed without multiplying the full mass fraction by density (`src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp:155`, `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp:172`). The expression effectively applies `*rho` only to the sinusoidal term, yielding incorrect (and for the right boundary, dramatically too large) scalar partial density.
- `QuokkaSimulation<ShocktubeProblem>::refineGrid(...)` (`reviewed`/`partial`): density-gradient tagging criterion reviewed.
- `QuokkaSimulation<ShocktubeProblem>::computeAfterTimestep()` (`reviewed`/`partial`): scalar conservation diagnostic accumulation reviewed; no additional confirmed bug in inspected path.
- `problem_main()` (`reviewed`): simulation run, conservation status handling, and optional plotting reviewed.

### `src/problems/HydroContact/testHydroContact.cpp`
- `ContactProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ContactProblem>` / `Physics_Traits<ContactProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<ContactProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): contact discontinuity IC setup reviewed.
- `QuokkaSimulation<ContactProblem>::computeReferenceSolution(...)` (`finding`, plotting-only): in the optional plotting block, vectors are pre-sized to `nx` and then appended with `push_back` inside the loop (`src/problems/HydroContact/testHydroContact.cpp:140-146`, `src/problems/HydroContact/testHydroContact.cpp:150`, `src/problems/HydroContact/testHydroContact.cpp:159-173`). This doubles vector lengths and prepends default-initialized zeros, corrupting plotted data (test numerics unaffected).
- `QuokkaSimulation<ContactProblem>::computeReferenceSolution(...)` (`reviewed`): reference fill and optional plotting path reviewed beyond the plotting-vector bug.
- `problem_main()` (`reviewed`): stationary contact test setup/evolve and exact-zero error check reviewed.

### `src/problems/HydroVacuum/testHydroVacuum.cpp`
- `ShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ShocktubeProblem>` / `Physics_Traits<ShocktubeProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): symmetric expansion-into-vacuum IC setup reviewed.
- `AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant left/right moving-state Dirichlet BC setup reviewed.
- `QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): Toro exact-solution read/interpolation, reference fill, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simulation run and error tolerance check reviewed.

### `src/problems/HydroWave/testHydroWave.cpp`
- `WaveProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<WaveProblem>` / `Physics_Traits<WaveProblem>` specializations (`reviewed`): trait constants reviewed.
- `computeWaveSolution(...)` (`reviewed`): cell-averaged linear-wave IC construction via eigenvector perturbation reviewed.
- `QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): initializes all comps to zero then fills wave solution.
- `problem_main()` (`reviewed`/`partial`): simulation run, error norm computation, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp`
- `WaveProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<WaveProblem>` / `Physics_Traits<WaveProblem>` specializations (`reviewed`): trait constants reviewed.
- `computeWaveSolution(...)` (`reviewed`): same wave IC helper pattern as `HydroWave`, reviewed.
- `QuokkaSimulation<WaveProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + wave IC setup reviewed.
- `runWaveTest(int)` (`reviewed`/`partial`): Richardson test single-resolution harness (ParmParse setup, evolve, error norm) reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): Richardson parameter setup and driver invocation reviewed.

### `src/problems/HydroBlast2D/testHydroBlast2D.cpp`
- `BlastProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<BlastProblem>` / `Physics_Traits<BlastProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<BlastProblem>::setInitialConditionsOnGrid(...)` (`finding`): initializes `energy_index` but never initializes `internalEnergy_index` (`src/problems/HydroBlast2D/testHydroBlast2D.cpp:92-103`), and this routine does not zero-fill all components first. The dual-energy variable can remain garbage/undefined at startup.
- `QuokkaSimulation<BlastProblem>::refineGrid(...)` (`reviewed`/`partial`): pressure-gradient tagging criterion reviewed.
- `problem_main()` (`reviewed`): simulation setup/evolve orchestration reviewed.

### `src/problems/HydroBlast3D/testHydroBlast3D.cpp`
- `SedovProblem` (`reviewed`): empty tag type for trait specialization.
- Global `simulate_full_box`, `test_passes`, `rho`, `E_blast` (`reviewed`): configuration/test-status globals reviewed.
- `quokka::EOS_Traits<SedovProblem>` / `HydroSystem_Traits<SedovProblem>` / `Physics_Traits<SedovProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<SedovProblem>::preCalculateInitialConditions()` (`reviewed`): octant-energy scaling for Sedov setup reviewed.
- `QuokkaSimulation<SedovProblem>::setInitialConditionsOnGrid(...)` (`finding`): after zero-filling all components, it sets `energy_index` but never sets `internalEnergy_index` (`src/problems/HydroBlast3D/testHydroBlast3D.cpp:95-103`). The dual-energy field remains zero and is inconsistent with the deposited blast energy.
- `QuokkaSimulation<SedovProblem>::refineGrid(...)` (`reviewed`/`partial`): pressure-gradient tagging criterion reviewed.
- `QuokkaSimulation<SedovProblem>::computeAfterEvolve(...)` (`reviewed`/`partial`): energy/kinetic-energy diagnostic checks and pass/fail flag logic reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): BC selection, evolve, and status return reviewed.

### `src/problems/HydroHighMach/testHydroHighMach.cpp`
- `HighMachProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<HighMachProblem>` / `Physics_Traits<HighMachProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<HighMachProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): high-Mach sinusoidal velocity IC setup reviewed.
- `QuokkaSimulation<HighMachProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): reference-file read, interpolation, reference fill, CSV write, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simulation run and error tolerance check reviewed.

### `src/problems/HydroLeblanc/testHydroLeblanc.cpp`
- `ShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ShocktubeProblem>` / `Physics_Traits<ShocktubeProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): LeBlanc shocktube IC setup reviewed.
- `AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant left/right Dirichlet states reviewed.
- `QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): exact-solution read/interpolation, reference fill, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`finding`): computes custom `BCs_cc` (`src/problems/HydroLeblanc/testHydroLeblanc.cpp:355`) but constructs `QuokkaSimulation<ShocktubeProblem> sim;` without passing them (`src/problems/HydroLeblanc/testHydroLeblanc.cpp:357`). The intended custom BC configuration is ignored.
- `problem_main()` (`reviewed`): otherwise standard setup/evolve/error check path reviewed.

### `src/problems/HydroShuOsher/testHydroShuOsher.cpp`
- `ShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ShocktubeProblem>` / `Physics_Traits<ShocktubeProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): Shu-Osher IC setup reviewed.
- `AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left/right Dirichlet state setup reviewed.
- `QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): reference-file read/interpolation, reference fill, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`finding`): BC initialization loop writes x-direction BCs to `BCs_cc[0]` instead of `BCs_cc[n]` (`src/problems/HydroShuOsher/testHydroShuOsher.cpp:286-289`). Only component 0 gets x-boundary settings; other components may retain incorrect/default BCs.
- `problem_main()` (`reviewed`): remaining simulation setup/evolve/error check reviewed.

### `src/problems/HydroKelvinHelmholz/testHydroKelvinHelmholz.cpp`
- `KelvinHelmholzProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<KelvinHelmholzProblem>` / `HydroSystem_Traits<KelvinHelmholzProblem>` / `Physics_Traits<KelvinHelmholzProblem>` specializations (`reviewed`): trait constants and hydro settings reviewed.
- `QuokkaSimulation<KelvinHelmholzProblem>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): KH shear-layer IC and perturbation setup reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<KelvinHelmholzProblem>::refineGrid(...)` (`reviewed`/`partial`): density-gradient tagging criterion reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): standard setup/evolve driver reviewed.

### `src/problems/HydroQuirk/testHydroQuirk.cpp`
- Alias `using Real = amrex::Real` and problem constants `dl, ul, pl, dr, ur, pr, ishock_g` (`reviewed`): constants reviewed; see `ishock_g` finding below.
- `QuirkProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<QuirkProblem>` / `HydroSystem_Traits<QuirkProblem>` / `Physics_Traits<QuirkProblem>` specializations (`reviewed`): trait constants and hydro settings reviewed.
- `QuokkaSimulation<QuirkProblem>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): Quirk odd-even IC setup and perturbed shock-row initialization reviewed.
- `getDeltaEntropyVector()` (`reviewed`): static accumulator accessor reviewed.
- `QuokkaSimulation<QuirkProblem>::computeAfterTimestep()` (`finding`): entropy-jump diagnostic samples at `ilo = ishock_g` (`src/problems/HydroQuirk/testHydroQuirk.cpp:145`), but `ishock_g` is a compile-time constant `0` (`src/problems/HydroQuirk/testHydroQuirk.cpp:69`) and is never updated from the computed shock index in `setInitialConditionsOnGrid(...)` (`src/problems/HydroQuirk/testHydroQuirk.cpp:79-84`). The carbuncle diagnostic can monitor the wrong x-location.
- `QuokkaSimulation<QuirkProblem>::computeAfterTimestep()` (`reviewed`): rank-0 box lookup, device entropy evaluation, and host accumulation path otherwise reviewed.
- `QuokkaSimulation<QuirkProblem>::computeAfterEvolve(...)` (`reviewed`/`partial`): post-run carbuncle threshold check and abort path reviewed.
- `AMRSimulation<QuirkProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant left/right Dirichlet states for Quirk test reviewed.
- `problem_main()` (`reviewed`): simulation setup/evolve driver reviewed.

### `src/problems/HydroRichtmeyerMeshkov/testHydroRichtmeyerMeshkov.cpp`
- `RichtmeyerMeshkovProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<RichtmeyerMeshkovProblem>` / `HydroSystem_Traits<RichtmeyerMeshkovProblem>` / `Physics_Traits<RichtmeyerMeshkovProblem>` specializations (`reviewed`): trait constants and hydro settings reviewed.
- `QuokkaSimulation<RichtmeyerMeshkovProblem>::computeAfterTimestep()` (`reviewed`/`partial`): rank-0 symmetry-check diagnostic (domain gather + x/y diagonal comparison) reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<RichtmeyerMeshkovProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): symmetric diagonal discontinuity IC setup reviewed.
- `problem_main()` (`reviewed`): reflecting-BC setup and simulation driver reviewed.

### `src/problems/HydroSMS/testHydroSMS.cpp`
- `ShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ShocktubeProblem>` / `Physics_Traits<ShocktubeProblem>` specializations (`reviewed`): trait constants reviewed.
- `QuokkaSimulation<ShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): shocktube IC setup with explicit zero-fill and dual-energy initialization reviewed.
- `AMRSimulation<ShocktubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant left/right Dirichlet state setup reviewed.
- `QuokkaSimulation<ShocktubeProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): piecewise exact-state construction, reference MultiFab fill, `fextract` analysis, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`/`partial`): setup/evolve/error-threshold path reviewed; comments on integrator/reconstruction order appear stale but no runtime bug confirmed from inspected code.

### `src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp`
- `HydrostaticAtmosphereProblem` (`reviewed`): empty tag type for trait specialization.
- `SimulationData<HydrostaticAtmosphereProblem>` / `quokka::EOS_Traits<HydrostaticAtmosphereProblem>` / `Physics_Traits<HydrostaticAtmosphereProblem>` specializations (`reviewed`): user-data and trait declarations reviewed.
- Constants/globals `kTgasInit`, `kRhoInitFactor`, `g_base_density_floor`, `g_scale_height` (`reviewed`): test configuration globals reviewed.
- `AMRSimulation<HydrostaticAtmosphereProblem>::setCustomBoundaryConditions(...)` (`reviewed`): exponential-atmosphere ghost-cell fill helper reviewed.
- `QuokkaSimulation<HydrostaticAtmosphereProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): hydrostatic IC setup with zero-fill + dual-energy initialization reviewed.
- `QuokkaSimulation<HydrostaticAtmosphereProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): parser/non-parser density-floor reference fill paths reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): parameter validation, managed-global setup, fixup invocation, and density-floor error check reviewed.

### `src/problems/BrioWuShockTube/testBrioWuShockTube.cpp`
- `MHDShocktubeProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<MHDShocktubeProblem>` / `Physics_Traits<MHDShocktubeProblem>` specializations (`reviewed`): trait constants and enabled-physics flags reviewed.
- Shock-state constants `rho_L`, `P_L`, `rho_R`, `P_R`, `Bx`, `By_L`, `By_R`, `Bz` (`reviewed`): test constants reviewed.
- `QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): cell-centered Brio-Wu IC setup (including dual-energy initialization) reviewed.
- `QuokkaSimulation<MHDShocktubeProblem>::setInitialConditionsOnGridFaceVars(...)` (`finding`, IC alignment): uses `x1_L = prob_lo[0] + i * dx[0]` for all face directions (`src/problems/BrioWuShockTube/testBrioWuShockTube.cpp:124`) to choose the left/right `B_y` state. For `dir == y` (where x-index is cell-centered, not nodal), this shifts the tangential-field discontinuity by half a cell.
- `AMRSimulation<MHDShocktubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant left/right MHD state Dirichlet BCs for cell-centered variables reviewed.
- `AMRSimulation<MHDShocktubeProblem>::setCustomBoundaryConditionsFaceVar<dir>(...)` (`reviewed`): constant face-centered MHD BC helper dispatch reviewed.
- `QuokkaSimulation<MHDShocktubeProblem>::refineGrid(...)` (`reviewed`/`partial`): density-gradient tagging criterion reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simple setup/evolve driver reviewed.

### `src/problems/CurrentSheet/testCurrentSheet.cpp`
- `CurrentSheet` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<CurrentSheet>` / `Physics_Traits<CurrentSheet>` specializations (`reviewed`): trait constants and enabled-physics flags reviewed.
- Constants `gamma_gas`, `beta`, `A`, `rho0`, `P0` (`reviewed`): test constants reviewed.
- `QuokkaSimulation<CurrentSheet>::setInitialConditionsOnGrid(...)` (`reviewed`): current-sheet hydrodynamic state initialization reviewed.
- `QuokkaSimulation<CurrentSheet>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`/`partial`): face-centered magnetic-field initialization reviewed; comment notes observed nonzero `Bx`, but no additional root cause confirmed in this function alone.
- `problem_main()` (`finding`): constructs periodic face-centered BC records `BCs_fc` (`src/problems/CurrentSheet/testCurrentSheet.cpp:111-118`) but then instantiates `QuokkaSimulation<CurrentSheet> sim;` without passing them (`src/problems/CurrentSheet/testCurrentSheet.cpp:120`). The custom face BC configuration is ignored.
- `problem_main()` (`reviewed`): remaining setup/evolve driver reviewed.

### `src/problems/FieldLoop/testFieldLoop.cpp`
- `FieldLoop` (`reviewed`): empty tag type for trait specialization.
- `RefineOn` enum (`reviewed`): refinement-mode selector reviewed.
- `quokka::EOS_Traits<FieldLoop>` / `Physics_Traits<FieldLoop>` specializations (`reviewed`): trait constants and enabled-physics flags reviewed.
- Constants `A`, `R_0` (`reviewed`): field-loop IC constants reviewed.
- `QuokkaSimulation<FieldLoop>::setInitialConditionsOnGrid(...)` (`finding`): sets `x3Momentum = rho0 * vz` with `vz = 1.0` (`src/problems/FieldLoop/testFieldLoop.cpp:71`, `:90`) but computes `Ekin` using only `vx^2 + vy^2` (`src/problems/FieldLoop/testFieldLoop.cpp:73`) before forming total energy (`src/problems/FieldLoop/testFieldLoop.cpp:92`). The conservative total energy is inconsistent with the initialized momentum state.
- `QuokkaSimulation<FieldLoop>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): vector-potential-based face-field initialization reviewed.
- `QuokkaSimulation<FieldLoop>::refineGrid(...)` (`finding`): region-based refinement computes normalized coordinates as `((i+0.5)*dx)/ (phi-plo)` (`src/problems/FieldLoop/testFieldLoop.cpp:148-149`) without subtracting `ProbLo()`. If the domain lower bound is not zero, the refinement window is shifted/misplaced.
- `QuokkaSimulation<FieldLoop>::refineGrid(...)` (`reviewed`): magnetic-energy tagging path reviewed; no additional confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<FieldLoop>::ComputeDerivedVar(...)` (`reviewed`/`partial`): `magnetic_divergence` derived variable computation reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simple setup/evolve driver reviewed.

### `src/problems/OrszagTang/testOrszagTang.cpp`
- `OrszagTang` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<OrszagTang>` / `Physics_Traits<OrszagTang>` specializations (`reviewed`): trait constants and enabled-physics flags reviewed.
- Constant `B0` and helpers `A_z(...)`, `B_x(...)`, `B_y(...)` (`reviewed`): vector-potential and magnetic-field helper functions reviewed.
- `QuokkaSimulation<OrszagTang>::setInitialConditionsOnGrid(...)` (`reviewed`): Orszag-Tang cell-centered IC setup (including magnetic-energy contribution) reviewed.
- `QuokkaSimulation<OrszagTang>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): face-centered magnetic-field initialization reviewed.
- `problem_main()` (`reviewed`): simple setup/evolve driver reviewed.

### `src/problems/MHDBlast/testMHDBlast.cpp`
- `MHDBlast` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<MHDBlast>` / `HydroSystem_Traits<MHDBlast>` / `Physics_Traits<MHDBlast>` specializations (`reviewed`): trait constants and hydro/MHD settings reviewed.
- `QuokkaSimulation<MHDBlast>::setInitialConditionsOnGrid(...)` (`reviewed`): 3D blast IC and dual-energy initialization reviewed.
- `QuokkaSimulation<MHDBlast>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): uniform face-centered magnetic-field initialization reviewed.
- `QuokkaSimulation<MHDBlast>::refineGrid(...)` (`reviewed`/`partial`): pressure-gradient tagging criterion using MHD pressure helper reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<MHDBlast>::ComputeDerivedVar(...)` (`reviewed`/`partial`): `magnetic_divergence` derived-variable path reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): simple setup/evolve driver reviewed.

### `src/problems/MHDQuirk/testMHDQuirk.cpp`
- Alias `using Real = amrex::Real` and problem constants `dl, ul, pl, dr, ur, pr, ishock_g` (`reviewed`): constants reviewed; see `ishock_g` finding below.
- `MHDQuirk` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<MHDQuirk>` / `HydroSystem_Traits<MHDQuirk>` / `Physics_Traits<MHDQuirk>` specializations (`reviewed`): trait constants and hydro/MHD settings reviewed.
- `QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): zero-magnetic-field face initialization reviewed.
- `QuokkaSimulation<MHDQuirk>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): Quirk odd-even IC setup and perturbed shock-row initialization reviewed.
- `getDeltaEntropyVector()` (`reviewed`): static accumulator accessor reviewed.
- `QuokkaSimulation<MHDQuirk>::computeAfterTimestep()` (`finding`): entropy-jump diagnostic samples at `ilo = ishock_g` (`src/problems/MHDQuirk/testMHDQuirk.cpp:160`), but `ishock_g` is a compile-time constant `0` (`src/problems/MHDQuirk/testMHDQuirk.cpp:69`) and is never updated from the computed shock index in `setInitialConditionsOnGrid(...)` (`src/problems/MHDQuirk/testMHDQuirk.cpp:94-100`). The carbuncle diagnostic can monitor the wrong x-location.
- `QuokkaSimulation<MHDQuirk>::computeAfterTimestep()` (`reviewed`): rank-0 box lookup, MHD pressure diagnostic, and host accumulation path otherwise reviewed.
- `QuokkaSimulation<MHDQuirk>::computeAfterEvolve(...)` (`reviewed`/`partial`): post-run carbuncle threshold check and abort path reviewed.
- `AMRSimulation<MHDQuirk>::setCustomBoundaryConditions(...)` (`reviewed`): constant left/right Dirichlet BCs for cell-centered variables reviewed.
- `problem_main()` (`reviewed`): BC setup (cell + face), simulation setup, and evolve driver reviewed.

### `src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp`
- `MHDBitwiseICs` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<MHDBitwiseICs>` / `Physics_Traits<MHDBitwiseICs>` specializations (`reviewed`): trait constants and enabled-physics flags reviewed.
- `computeWaveSolution(...)` (`finding`, portability/robustness): unconditionally reads `prob_lo[2]` and `dx[2]` (`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp:56-58`) even though the file’s explicit 3D requirement is only enforced later inside `verifyPeriodicBCs(...)` (`src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp:159-160`). In non-3D builds, initialization/reference paths can hit out-of-bounds access before the guard runs.
- `computeWaveSolution(...)` (`reviewed`/`partial`): deterministic unique-value IC/reference helper logic otherwise reviewed.
- `QuokkaSimulation<MHDBitwiseICs>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + CC IC wrapper reviewed.
- `QuokkaSimulation<MHDBitwiseICs>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): zero-fill + FC IC wrapper reviewed.
- `QuokkaSimulation<MHDBitwiseICs>::computeReferenceSolution(...)` (`reviewed`): CC reference fill wrapper reviewed.
- `QuokkaSimulation<MHDBitwiseICs>::computeReferenceSolution_fc(...)` (`reviewed`): FC reference fill wrapper reviewed.
- `verifyPeriodicBCs(...)` (`reviewed`/`partial`): host mirror creation, periodic wrapping checks, and mismatch reporting reviewed; no additional confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): initialization, periodic ghost-cell verification, and status return path reviewed.

### `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp`
- `MHDBalsaraVortex` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<MHDBalsaraVortex>` / `Physics_Traits<MHDBalsaraVortex>` specializations (`reviewed`): trait constants and enabled-physics flags reviewed.
- Global constants/managed parameters (`gamma_gas`, `bg_density`, `bg_pressure`, `sound_speed`, `vortex_*`) (`reviewed`): vortex configuration state reviewed.
- Helpers `computeRadiusSq(...)`, `computeRadialProfile(...)`, `Az(...)` (`reviewed`): analytic vortex profile helpers reviewed.
- `computeVortexSolution(...)` (`reviewed`/`partial`): CC/FC analytic vortex state construction reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + CC vortex IC wrapper reviewed.
- `QuokkaSimulation<MHDBalsaraVortex>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): zero-fill + FC vortex IC wrapper reviewed.
- `QuokkaSimulation<MHDBalsaraVortex>::computeReferenceSolution(...)` (`reviewed`): CC reference fill wrapper reviewed.
- `QuokkaSimulation<MHDBalsaraVortex>::computeReferenceSolution_fc(...)` (`reviewed`): FC reference fill wrapper reviewed.
- `problem_main()` (`finding`, robustness): `stop_time` calculation divides by `vortex_u_magn = vortex_Mach * sound_speed` (`src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp:235`, `:264`, `:268`) without validating `vortex_Mach > 0`. A user input of `setup.vortex_Mach = 0` yields division by zero (`inf`/`nan` stop time).
- `problem_main()` (`reviewed`): parameter parsing, BC setup, drift/orbit timing logic, evolve, and error-threshold check reviewed.

### `src/problems/RandomBlast/testRandomBlast.cpp`
- `RandomBlast` (`reviewed`): empty tag type for trait specialization.
- Constants `m_H`, `seconds_per_year`, `cloudy_H_mass_fraction` (`reviewed`): physical/test constants reviewed.
- `Physics_Traits<RandomBlast>` / `quokka::EOS_Traits<RandomBlast>` / `Particle_Traits<RandomBlast>` / `SimulationData<RandomBlast>` specializations (`reviewed`): trait settings and user-data schema reviewed.
- `QuokkaSimulation<RandomBlast>::setInitialConditionsOnGrid(...)` (`reviewed`): ambient hydro state initialization with passive scalar + dual-energy setup reviewed.
- `QuokkaSimulation<RandomBlast>::createInitialStochasticStellarPopParticles()` (`reviewed`/`partial`): particle ASCII load and per-particle stage/velocity adjustments reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<RandomBlast>::computeAfterTimestep()` (`reviewed`): cumulative SN-count logging hook reviewed.
- `QuokkaSimulation<RandomBlast>::ComputeDerivedVar(...)` (`reviewed`): `temperature` derived-variable branch reviewed.
- `problem_main()` (`finding`, portability): unconditionally extracts a z-axis slice with `fextract(..., 2, ...)` (`src/problems/RandomBlast/testRandomBlast.cpp:203`) without a dimension guard or 3D-only assertion. In 1D/2D builds, this driver is not dimension-safe.
- `problem_main()` (`reviewed`): parameter parsing, evolve path, SN-count printout, and optional z-temperature plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/ShockCloud/testShockCloud.cpp`
- Alias `using amrex::Real`, constants (`seconds_in_year`, `parsec_in_cm`, `solarmass_in_g`, `keV_in_ergs`, `m_H`) (`reviewed`): physical conversion constants reviewed.
- `ShockCloud` (`reviewed`): empty tag type for trait specialization.
- `Physics_Traits<ShockCloud>` / `quokka::EOS_Traits<ShockCloud>` specializations (`reviewed`): trait constants and passive-scalar configuration reviewed.
- Global problem/runtime state (`sharp_cloud_edge`, `do_frame_shift`, `rho0`, `rho1`, `P0`, `R_cloud`, `cloud_relpos_x`, `shock_crossing_time`, `rho_wind`, `v_wind`, `P_wind`, `delta_vx`) (`reviewed`): mutable globals used across ICs/BCs/statistics reviewed.
- `QuokkaSimulation<ShockCloud>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): cloud/background IC construction, smoothed-edge option, passive-scalar initialization, and gas-energy initialization reviewed; no confirmed bug in inspected implementation this pass.
- `AMRSimulation<ShockCloud>::setCustomBoundaryConditions(...)` (`reviewed`/`partial`): time-dependent inflow/outflow NSCBC and early-time Dirichlet boundary path reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<ShockCloud>::computeAfterTimestep()` (`finding`, robustness): frame-shift update computes `vx_cm = xmom / cloud_mass` (`src/problems/ShockCloud/testShockCloud.cpp:232`) without guarding `cloud_mass <= 0`. If the tracked cloud mass becomes zero/underflows, the frame-shift state and subsequent momentum/energy updates become invalid (`inf`/`nan`).
- `QuokkaSimulation<ShockCloud>::computeAfterTimestep()` (`reviewed`): metadata accumulation, velocity-shift logging, and momentum/energy Galilean transform loop reviewed.
- `QuokkaSimulation<ShockCloud>::ComputeDerivedVar(...)` (`finding`, robustness): `cloud_fraction` branch computes `rho_cloud / (rho_cloud + rho_bg)` without a zero/positivity guard (`src/problems/ShockCloud/testShockCloud.cpp:425`). If both partial densities vanish (e.g., pathological/ghost-cell state), this yields `nan`.
- `QuokkaSimulation<ShockCloud>::ComputeDerivedVar(...)` (`reviewed`): reviewed branches for `temperature`, `c_s`, `nH*`, `nH_residual`, `pressure`, `entropy`, `mass`, `cloud_fraction`, `cooling_length`, `lab_velocity_x`, and `velocity_mag`; no additional confirmed bug beyond the explicit `cloud_fraction` denominator issue this pass.
- `ComputeCellTempResampled(...)` (`reviewed`): resampled-cooling temperature helper reviewed.
- `QuokkaSimulation<ShockCloud>::ComputeStatistics()` (`finding`, robustness): cloud-fraction statistics recompute `C_frac = rho_cloud / (rho_cloud + rho_bg)` without guarding the denominator (`src/problems/ShockCloud/testShockCloud.cpp:636`, `:645`), allowing `nan` statistics for zero partial-density states.
- `QuokkaSimulation<ShockCloud>::ComputeStatistics()` (`reviewed`): reviewed time/offset stats, mass integrals, temperature-threshold cloud-mass integrals, and scalar/fraction-threshold metrics.
- `QuokkaSimulation<ShockCloud>::refineGrid(...)` (`reviewed`/`partial`): cooling-length-based AMR tagging reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`finding`, robustness): shock-jump setup divides by `M0` when computing `v_wind` (`src/problems/ShockCloud/testShockCloud.cpp:765`) and uses `v_shock = M0 * x4` downstream (`src/problems/ShockCloud/testShockCloud.cpp:767`, `:781`, `:786`) without validating `Mach_shock > 0`. Zero/invalid `Mach_shock` inputs can produce divide-by-zero/`nan` setup values.
- `problem_main()` (`reviewed`): parameter parsing, cooling-table-dependent setup, shock-jump calculations, metadata initialization, and evolve path reviewed.

### `src/problems/Advection/testAdvection.cpp`
- `SawtoothProblem` (`reviewed`): empty tag type for trait specialization.
- `Physics_Traits<SawtoothProblem>` specialization (`reviewed`): advection-only trait configuration reviewed.
- `ComputeExactSolution(...)` (`reviewed`): periodic sawtooth exact-profile helper reviewed.
- `AdvectionSimulation<SawtoothProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): exact-solution IC fill wrapper reviewed.
- `AdvectionSimulation<SawtoothProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): reference MultiFab fill and optional plotting path reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`/`partial`): advection setup, evolve, and error-threshold check reviewed; note `advectionVy_` assignment appears nonessential for this test but no confirmed runtime bug from inspected code.

### `src/problems/Advection2D/testAdvection2D.cpp`
- Alias `using amrex::Real` (`reviewed`): local type alias reviewed.
- `SquareProblem` (`reviewed`): empty tag type for trait specialization.
- `Physics_Traits<SquareProblem>` specialization (`reviewed`): advection-only trait configuration reviewed.
- `exactSolutionAtIndex(...)` (`reviewed`): square-pulse exact profile helper reviewed.
- `AdvectionSimulation<SquareProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): IC fill wrapper reviewed.
- `AdvectionSimulation<SquareProblem>::computeReferenceSolution(...)` (`reviewed`): reference fill path reviewed.
- `AdvectionSimulation<SquareProblem>::refineGrid(...)` (`reviewed`/`partial`): gradient-based AMR tagging criterion reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): setup/evolve/error-threshold path reviewed.

### `src/problems/AdvectionSemiellipse/testAdvectionSemiellipse.cpp`
- `SemiellipseProblem` (`reviewed`): empty tag type for trait specialization.
- `Physics_Traits<SemiellipseProblem>` specialization (`reviewed`): advection-only trait configuration reviewed.
- `ComputeExactSolution(...)` (`reviewed`): semiellipse profile helper reviewed.
- `AdvectionSimulation<SemiellipseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): IC fill wrapper reviewed.
- `AdvectionSimulation<SemiellipseProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): reference fill and optional plotting path reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): setup/evolve/error-threshold path reviewed.

### `src/problems/FCQuantities/testFCQuantities.cpp`
- `FCQuantities` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<FCQuantities>` / `Physics_Traits<FCQuantities>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants `rho0`, `P0`, `v0`, `amp` (`reviewed`): test constants reviewed.
- `computeWaveSolution(...)` (`reviewed`): CC hydrodynamic perturbation initialization helper reviewed.
- `QuokkaSimulation<FCQuantities>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + CC IC wrapper reviewed.
- `QuokkaSimulation<FCQuantities>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`/`partial`): divergence-free face-field initialization from nodal potential reviewed; no confirmed bug in inspected implementation this pass.
- `setAmrNCell(...)` (`reviewed`): helper for injecting `amr.n_cell` ParmParse values reviewed.
- `setPlotfileParams(...)` (`reviewed`): helper for plotfile ParmParse values reviewed.
- `checkDivFreeRestart(...)` (`reviewed`/`partial`): divergence check after restart refinement reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`/`partial`): pre/post-restart setup and divergence-check orchestration reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/FastWave/testFastWave.cpp`
- `FastWave` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<FastWave>` / `Physics_Traits<FastWave>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants (`sound_speed`, `gamma_gas`, background states, wave parameters, `omega`) (`reviewed`): analytic test constants reviewed.
- `computeMagneticVectorPotential_x/y/z(...)` (`reviewed`): vector-potential helper functions reviewed.
- `computeWaveSolution(...)` (`reviewed`/`partial`): CC/FC fast-wave analytic solution construction reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<FastWave>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + CC IC wrapper reviewed.
- `QuokkaSimulation<FastWave>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): zero-fill + FC IC wrapper reviewed.
- `QuokkaSimulation<FastWave>::computeReferenceSolution(...)` (`reviewed`): CC reference fill wrapper reviewed.
- `QuokkaSimulation<FastWave>::computeReferenceSolution_fc(...)` (`reviewed`): FC reference fill wrapper reviewed.
- `problem_main()` (`reviewed`/`partial`): evolve path and custom combined error-norm calculation reviewed; zero-reference components are handled robustly in inspected logic.

### `src/problems/AlfvenWaveCircular/testAlfvenWaveCircular.cpp`
- `AlfvenWaveCircular` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<AlfvenWaveCircular>` / `Physics_Traits<AlfvenWaveCircular>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants (`sound_speed`, `gamma_gas`, wave/background parameters, `omega`) (`reviewed`): analytic test constants reviewed.
- `computeMagneticVectorPotential_x/y/z(...)` (`reviewed`): vector-potential helper functions reviewed.
- `computeWaveSolution(...)` (`reviewed`/`partial`): CC/FC circular Alfvén-wave analytic solution construction reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<AlfvenWaveCircular>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + CC IC wrapper reviewed.
- `QuokkaSimulation<AlfvenWaveCircular>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): zero-fill + FC IC wrapper reviewed.
- `QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution(...)` (`reviewed`): CC reference fill wrapper reviewed.
- `QuokkaSimulation<AlfvenWaveCircular>::computeReferenceSolution_fc(...)` (`reviewed`): FC reference fill wrapper reviewed.
- `problem_main()` (`reviewed`): setup/evolve/error-threshold path reviewed.

### `src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp`
- `AlfvenWaveLinear` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<AlfvenWaveLinear>` / `Physics_Traits<AlfvenWaveLinear>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants and GPU-managed wave-frame globals (`sound_speed`, `gamma_gas`, `bg_*`, `delta_b_magn`, `alfven_speed`, `angle_between_k_b0_rad`, `k_*`, basis vectors) (`reviewed`): analytic setup state reviewed.
- Vector helpers `computeMagnitude(...)`, `computeDotProduct(...)`, `computeCrossProduct(...)`, `normalizeVector(...)` (`reviewed`): geometry helper functions reviewed.
- Rotation helpers `rotatePRF2MRF(...)`, `rotateMRF2PRF(...)` (`reviewed`): basis-rotation helpers reviewed.
- Vector-potential helpers `computeVectorPotentialComponent_prf(...)`, `Ax_prf(...)`, `Ay_prf(...)`, `Az_prf(...)` (`reviewed`): analytic magnetic vector potential construction reviewed.
- `computeWaveSolution(...)` (`finding`, portability): unconditionally accesses `prob_lo[2]` and `dx[2]` (`src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp:221`, `:226`, and FC stencil terms `:277-286`) without a 3D-only guard. In non-3D builds this helper is not dimension-safe.
- `computeWaveSolution(...)` (`reviewed`): CC/FC linear Alfvén-wave state construction reviewed; no additional confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGrid(...)` (`reviewed`): zero-fill + CC IC wrapper reviewed.
- `QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): zero-fill + FC IC wrapper reviewed.
- `QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution(...)` (`reviewed`): CC reference fill wrapper reviewed.
- `QuokkaSimulation<AlfvenWaveLinear>::computeReferenceSolution_fc(...)` (`reviewed`): FC reference fill wrapper reviewed.
- `problem_main()` (`reviewed`/`partial`): parameter parsing, basis construction, evolve path, and error-threshold check reviewed.

### `src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp`
- `AlfvenWaveLinear` (`reviewed`): problem tag type reused for convergence harness.
- `quokka::EOS_Traits<AlfvenWaveLinear>` / `Physics_Traits<AlfvenWaveLinear>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants/global GPU-managed frame state and helper families (`computeMagnitude`, `computeDotProduct`, `computeCrossProduct`, `normalizeVector`, `rotatePRF2MRF`, `rotateMRF2PRF`, vector-potential helpers) (`reviewed`/`partial`): reviewed as convergence-file analog of `testAlfvenWaveLinear.cpp`; no additional confirmed bug beyond those noted below.
- `computeWaveSolution(...)` (`finding`, portability): same unguarded 3D-only accesses (`prob_lo[2]`, `dx[2]`) in CC/FC wave helper (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:225`, `:230`, `:277-282`), which is not dimension-safe for non-3D builds.
- `QuokkaSimulation<AlfvenWaveLinear>::setInitialConditionsOnGrid(...)` / `setInitialConditionsOnGridFaceVars(...)` / `computeReferenceSolution(...)` / `computeReferenceSolution_fc(...)` (`reviewed`): wrapper IC/reference fill functions reviewed.
- `runWaveTest(int)` (`finding`, robustness): computes `cA = alfven_speed * abs(cos(angle_between_k_b0_rad))` (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:381`) and then `max_time = wavelength / cA` (`src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp:400`) without guarding `cA > 0`. A perpendicular setup (`angle_between_k_b0 = 90 deg`) yields division by zero.
- `runWaveTest(int)` (`reviewed`): wavevector/basis setup, AMReX ParmParse grid injection, simulation run, and error return reviewed.
- `problem_main()` (`reviewed`): Richardson convergence harness setup reviewed.

### `src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp`
- `EntropyWaveLinear` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<EntropyWaveLinear>` / `Physics_Traits<EntropyWaveLinear>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants/global GPU-managed frame state and helper families (`adv_speed`, `gamma_gas`, `bg_*`, `computeMagnitude`, `computeDotProduct`, `computeCrossProduct`, `normalizeVector`, `rotatePRF2MRF`, `rotateMRF2PRF`, vector-potential helpers) (`reviewed`/`partial`): analytic entropy-wave setup and geometry helpers reviewed.
- `computeWaveSolution(...)` (`finding`, portability): unguarded accesses to `prob_lo[2]` and `dx[2]` in CC/FC branches (`src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp:168`, `:173`, `:213-222`) make the helper non-portable to non-3D builds.
- `computeWaveSolution(...)` (`reviewed`): entropy-mode CC/FC analytic state construction reviewed; no additional confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<EntropyWaveLinear>::setInitialConditionsOnGrid(...)` / `setInitialConditionsOnGridFaceVars(...)` / `computeReferenceSolution(...)` / `computeReferenceSolution_fc(...)` (`reviewed`): wrapper IC/reference fill functions reviewed.
- `runWaveTest(int)` (`reviewed`/`partial`): wavevector/basis setup, AMReX ParmParse grid injection, simulation run, and error return reviewed; no confirmed bug in inspected implementation this pass (`adv_speed` is compile-time positive).
- `problem_main()` (`reviewed`): Richardson convergence harness setup reviewed.

### `src/problems/FastWaveConvergence/testFastWaveConvergence.cpp`
- `FastWaveConvergence` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<FastWaveConvergence>` / `Physics_Traits<FastWaveConvergence>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants/global GPU-managed frame state and helper families (`gamma_gas`, `b0_magn`, `computeMagnitude`, `computeDotProduct`, `computeCrossProduct`, `normalizeVector`, rotations, vector-potential helpers) (`reviewed`/`partial`): analytic fast-wave setup and geometry helpers reviewed.
- `computeWaveSolution(...)` (`finding`, portability): unguarded 3D-only accesses to `prob_lo[2]`/`dx[2]` in CC/FC branches (`src/problems/FastWaveConvergence/testFastWaveConvergence.cpp:240`, `:248`, `:325-334`) make the helper non-portable to non-3D builds.
- `computeWaveSolution(...)` (`reviewed`): fast-wave analytic CC/FC construction reviewed; no additional confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<FastWaveConvergence>::setInitialConditionsOnGrid(...)` / `setInitialConditionsOnGridFaceVars(...)` / `computeReferenceSolution(...)` / `computeReferenceSolution_fc(...)` (`reviewed`): wrapper IC/reference fill functions reviewed.
- `runWaveTest(int)` (`reviewed`/`partial`): phase-speed calculation, wavevector/basis setup, AMReX ParmParse grid injection, simulation run, and error return reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): Richardson convergence harness setup reviewed.

### `src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp`
- `SlowWaveConvergence` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<SlowWaveConvergence>` / `Physics_Traits<SlowWaveConvergence>` specializations (`reviewed`): trait constants and hydro/MHD configuration reviewed.
- Constants/global GPU-managed frame state and helper families (`gamma_gas`, `b0_magn`, `computeMagnitude`, `computeDotProduct`, `computeCrossProduct`, `normalizeVector`, rotations, vector-potential helpers) (`reviewed`/`partial`): analytic slow-wave setup and geometry helpers reviewed.
- `computeWaveSolution(...)` (`finding`, portability): unguarded 3D-only accesses to `prob_lo[2]`/`dx[2]` in CC/FC branches (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:241`, `:249`, `:335-344`) make the helper non-portable to non-3D builds.
- `computeWaveSolution(...)` (`reviewed`): slow-wave analytic CC/FC construction reviewed; no additional confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<SlowWaveConvergence>::setInitialConditionsOnGrid(...)` / `setInitialConditionsOnGridFaceVars(...)` / `computeReferenceSolution(...)` / `computeReferenceSolution_fc(...)` (`reviewed`): wrapper IC/reference fill functions reviewed.
- `runWaveTest(int)` (`finding`, robustness): computes slow magnetosonic speed `cs` (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:444`) and then `max_time = wavelength / cs` (`src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp:463`) without guarding `cs > 0`. Perpendicular configurations can make `cs` approach or equal zero, causing division by zero/very large stop times.
- `runWaveTest(int)` (`reviewed`): wavevector/basis setup, AMReX ParmParse grid injection, simulation run, and error return reviewed.
- `problem_main()` (`reviewed`): Richardson convergence harness setup reviewed.

### `src/problems/NscbcChannel/testNscbcChannel.cpp`
- `Channel` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<Channel>` / `Physics_Traits<Channel>` specializations (`reviewed`): NSCBC channel physics configuration and scalar count reviewed.
- Global inflow/outflow/IC state (`rho0`, `u0`, `s0`, `Tgas0`, `P_outflow`, `u_inflow`, `v_inflow`, `w_inflow`, `s_inflow`) (`reviewed`): runtime-populated BC/IC globals reviewed.
- `QuokkaSimulation<Channel>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform channel IC fill reviewed.
- `AMRSimulation<Channel>::setCustomBoundaryConditions(...)` (`reviewed`): x-lower inflow / x-upper NSCBC outflow dispatch reviewed.
- `problem_main()` (`finding`, correctness): pressure diagnostic reconstructs `Eint` as `Egas - xmom^2/(2 rho)` (`src/problems/NscbcChannel/testNscbcChannel.cpp:172-180`), omitting `y/z` kinetic energy even though `v_inflow` and `w_inflow` are configurable. This biases diagnostic pressure/error checks when transverse inflow velocity is nonzero.
- `problem_main()` (`finding`, robustness): component-wise relative error accumulation divides by `U_k` with no zero guard (`src/problems/NscbcChannel/testNscbcChannel.cpp:193-208`). The passive-scalar reference magnitude is `|s_inflow|`, so `s_inflow = 0` yields division by zero in the error norm.
- `problem_main()` (`reviewed`): setup, evolve, slice extraction, plotting, and pass/fail threshold logic reviewed.

### `src/problems/NscbcVortex/testNscbcVortex.cpp`
- `Vortex` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<Vortex>` / `Physics_Traits<Vortex>` specializations (`reviewed`): NSCBC vortex physics configuration and scalar count reviewed.
- Anonymous-namespace globals/constants (`outflow_boundary_along_x_axis`, `G_vortex`, `T_ref`, `P_ref`, `u0`, `v0`, `w0`, `s0`) (`reviewed`): runtime-populated vortex/BC state reviewed.
- `QuokkaSimulation<Vortex>::setInitialConditionsOnGrid(...)` (`reviewed`): isentropic vortex IC construction reviewed; no confirmed bug in inspected implementation this pass.
- `AMRSimulation<Vortex>::setCustomBoundaryConditions(...)` (`reviewed`): outflow-axis dependent NSCBC outflow BC dispatch reviewed.
- `problem_main()` (`reviewed`): BC setup, parameter parsing, initialization, and evolve path reviewed.

### `src/problems/PassiveScalar/testPassiveScalar.cpp`
- `ScalarProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<ScalarProblem>` / `Physics_Traits<ScalarProblem>` specializations (`reviewed`): hydro + passive-scalar configuration reviewed.
- Constant `v_contact` (`reviewed`): analytic contact-wave speed reviewed.
- `QuokkaSimulation<ScalarProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): left/right contact + passive-scalar IC fill reviewed.
- `QuokkaSimulation<ScalarProblem>::computeReferenceSolution(...)` (`reviewed`/`partial`): exact contact solution fill, passive-scalar conservation check, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<ScalarProblem>::refineGrid(...)` (`reviewed`/`partial`): density-gradient tagging logic reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): initialize/evolve/error-threshold test harness reviewed.

### `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp`
- `BinaryOrbit` (`reviewed`): empty tag type for trait specialization.
- Static globals `do_split_particles`, `split_factor` (`reviewed`): runtime particle-splitting controls reviewed.
- `quokka::EOS_Traits<BinaryOrbit>` / `Particle_Traits<BinaryOrbit>` / `HydroSystem_Traits<BinaryOrbit>` / `Physics_Traits<BinaryOrbit>` specializations (`reviewed`): isothermal self-gravity + CIC particle configuration reviewed.
- `SimulationData<BinaryOrbit>` specialization (`reviewed`): per-step orbit diagnostics storage reviewed.
- `QuokkaSimulation<BinaryOrbit>::setInitialConditionsOnGrid(...)` (`reviewed`): ambient gas background initialization reviewed.
- `QuokkaSimulation<BinaryOrbit>::createInitialCICParticles()` (`reviewed`): ASCII particle load and optional split path reviewed.
- `QuokkaSimulation<BinaryOrbit>::ComputeDerivedVar(...)` (`reviewed`): derived gravitational potential output hook reviewed.
- `QuokkaSimulation<BinaryOrbit>::computeAfterTimestep()` (`reviewed`/`partial`): periodic particle-separation diagnostic collection reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`/`partial`): initialization, evolve, particle-count/error checks, and split/refactor tolerances reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/DustAdvection/testDustAdvection.cpp`
- `DustAdvection` (`reviewed`): empty tag type for trait specialization.
- Test constants (`initial_Egas`, `rho`, `v0`, `dust_v0`, Gaussian parameters `rho_bg`, `A`, `sigma`, `Lx`, `xc`) (`reviewed`): analytic setup constants reviewed (note: `rho` appears unused).
- `quokka::EOS_Traits<DustAdvection>` / `Physics_Traits<DustAdvection>` specializations (`reviewed`): dust-enabled hydro test configuration reviewed.
- `QuokkaSimulation<DustAdvection>::setInitialConditionsOnGrid(...)` (`finding`, AMR correctness): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustAdvection/testDustAdvection.cpp:59-60`) instead of `grid_elem.dx_` / `grid_elem.prob_lo_`. If this IC routine runs on refined levels, coordinates are computed with level-0 geometry and the Gaussian profile is misplaced/mis-scaled.
- `QuokkaSimulation<DustAdvection>::computeReferenceSolution(...)` (`reviewed`/`partial`): periodic-shift exact solution and optional plotting reviewed; no additional confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): test harness parameterization and error-threshold check reviewed.

### `src/problems/DustAdvection3D/testDustAdvection3D.cpp`
- `DustAdvection3D` (`reviewed`): empty tag type for trait specialization.
- Test constants (`initial_Egas`, `v0`, `dust_v0`, Gaussian parameters `rho_bg`, `A`, `sigma`, `xc/yc/zc`, `Lx/Ly/Lz`) (`reviewed`): analytic setup constants reviewed.
- `quokka::EOS_Traits<DustAdvection3D>` / `Physics_Traits<DustAdvection3D>` specializations (`reviewed`): 3D dust-enabled hydro test configuration reviewed.
- `QuokkaSimulation<DustAdvection3D>::setInitialConditionsOnGrid(...)` (`finding`, AMR correctness): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustAdvection3D/testDustAdvection3D.cpp:62-63`) instead of `grid_elem` geometry. Refined-level IC fills would use level-0 coordinates and produce incorrect 3D Gaussian placement/width.
- `QuokkaSimulation<DustAdvection3D>::computeReferenceSolution(...)` (`reviewed`/`partial`): 3D periodic-shift exact solution and axis-slice plotting reviewed; no additional confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`): test harness parameterization and error-threshold check reviewed.

### `src/problems/GravRadParticle3D/testGravRadParticle3D.cpp`
- `ParticleProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`nGroups_`, radiation/gas floors and initial states, `c`, `chat`, `kappa0`, `rho0`, `m_H`, `lum1`) (`reviewed`): test constants reviewed (`m_H` appears unused in this file).
- `quokka::EOS_Traits<ParticleProblem>` / `Particle_Traits<ParticleProblem>` / `Physics_Traits<ParticleProblem>` / `RadSystem_Traits<ParticleProblem>` specializations (`reviewed`): particle+radiation+self-gravity configuration reviewed.
- `QuokkaSimulation<ParticleProblem>::createInitialCICRadParticles()` / `createInitialCICParticles()` / `createInitialRadParticles()` (`reviewed`): ASCII particle initialization hooks reviewed.
- `RadSystem<ParticleProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `QuokkaSimulation<ParticleProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation background initialization reviewed.
- `problem_main()` (`reviewed`/`partial`): evolve path, total-radiation energy check, and particle-position validation reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/DustDamping/testDustDamping.cpp`
- Analytic-solution forward declarations `v_gas_analytic(...)`, `v_dust1_analytic(...)`, `v_dust2_analytic(...)`, `E_gas_analytic(...)` (`reviewed`): declarations match later implementations.
- `DustDamping` (`reviewed`): empty tag type for trait specialization.
- `SimulationData<DustDamping>` specialization (`reviewed`): time-series storage for gas/dust velocities and gas energy reviewed.
- `quokka::EOS_Traits<DustDamping>` / `Physics_Traits<DustDamping>` specializations (`reviewed`): two-dust-group damping test configuration reviewed.
- Constants (`V_COM`, `LAMBDA*`, `C_*`, `rho_dust*`, `TS*`, `OMEGA`, `P_INITIAL`, `rho`, `v0`, `Egas0`, `Egas0_internal`, `numDustVars`) (`reviewed`): analytic/test setup constants reviewed.
- `DustDrag<DustDamping>::ComputeReciprocalStoppingTime(...)` (`reviewed`): constant stopping-time model for two dust groups reviewed.
- `QuokkaSimulation<DustDamping>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+dust damping IC fill reviewed.
- `QuokkaSimulation<DustDamping>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction and logging reviewed; no confirmed bug in inspected implementation this pass.
- `analytic_velocity(...)`, `v_gas_analytic(...)`, `v_dust1_analytic(...)`, `v_dust2_analytic(...)` (`reviewed`): analytic velocity helpers reviewed.
- `E_gas_analytic(...)` (`reviewed`/`partial`): trapezoidal drag-heating integration helper reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`/`partial`): initialize/evolve, analytic comparison, relative-error checks, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/DustDampingIteration/testDustDampingIteration.cpp`
- `DustDampingWithCorrection` / `DustDampingWithoutCorrection` (`reviewed`): tag types for iterative-stopping-time comparison harness.
- `SimulationData<...>` specializations (`reviewed`): duplicated timeseries storage types reviewed.
- `quokka::EOS_Traits<DustDampingWithCorrection>` / `quokka::EOS_Traits<DustDampingWithoutCorrection>` (`reviewed`): EOS trait constants reviewed.
- Constants (`rho_dust1`, `rho_dust2`, `P_INITIAL`, `rho`, `v0`, `Egas0_*`, `Egas0_internal_*`, `numDustVars`, `dust_grain_radius`, `dust_grain_density`, supersonic-correction flags) (`reviewed`): test setup constants reviewed.
- `Physics_Traits<DustDampingWithCorrection>` / `Physics_Traits<DustDampingWithoutCorrection>` (`reviewed`): paired dust test configurations reviewed.
- `DustDrag<DustDampingWithCorrection>::ComputeReciprocalStoppingTime(...)` / `DustDrag<DustDampingWithoutCorrection>::ComputeReciprocalStoppingTime(...)` (`reviewed`): Kwok drag wrappers with/without supersonic correction reviewed.
- `QuokkaSimulation<DustDampingWithCorrection>::setInitialConditionsOnGrid(...)` / `QuokkaSimulation<DustDampingWithoutCorrection>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+dust IC fill routines reviewed.
- `QuokkaSimulation<DustDampingWithCorrection>::computeAfterTimestep()` / `QuokkaSimulation<DustDampingWithoutCorrection>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction and logging reviewed; no confirmed bug in inspected implementation this pass.
- `run_reference_simulation()` (`reviewed`/`partial`): fixed-small-`dt` reference run configuration and data capture reviewed.
- `run_iterative_with_correction()` / `run_iterative_without_correction()` (`reviewed`/`partial`): iterative run setup and data capture reviewed.
- `compute_relative_error(...)` (`reviewed`/`partial`): nearest-timepoint L1 relative error helper reviewed; guard for empty/zero reference sums present.
- `problem_main()` (`reviewed`/`partial`): orchestration of three runs, comparisons, pass/fail thresholds, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/DustSoundwave/testDustSoundwave.cpp`
- `DustSoundwave` (`reviewed`): empty tag type for trait specialization.
- Analytic constants (`A`, `rho_g0`, `rho_d0`, `omega_r`, `omega_i`, real/imag coefficients) (`reviewed`): linear-mode analytic parameters reviewed.
- `real_part_analytic(...)`, `v_gas_analytic(...)`, `rho_gas_analytic(...)`, `v_dust_analytic(...)`, `rho_dust_analytic(...)` (`reviewed`): analytic mode-amplitude helpers reviewed.
- `SimulationData<DustSoundwave>` specialization (`reviewed`): time-series storage for gas/dust densities and velocities reviewed.
- `quokka::EOS_Traits<DustSoundwave>` / `Physics_Traits<DustSoundwave>` specializations (`reviewed`): isothermal dust sound-wave configuration reviewed.
- Global `cs` (`reviewed`): alias to isothermal sound speed reviewed.
- `DustDrag<DustSoundwave>::ComputeReciprocalStoppingTime(...)` (`reviewed`): constant reciprocal stopping-time model reviewed.
- `QuokkaSimulation<DustSoundwave>::setInitialConditionsOnGrid(...)` (`finding`, AMR correctness): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustSoundwave/testDustSoundwave.cpp:121-122`) rather than `grid_elem.dx_` / `grid_elem.prob_lo_`. Refined-level IC fills would use level-0 geometry and mis-phase the wave.
- `QuokkaSimulation<DustSoundwave>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction reviewed; no confirmed bug in inspected implementation this pass.
- `problem_main()` (`reviewed`/`partial`): initialize/evolve, normalization against analytic solution, relative-error checks, and plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/DustyShock/testDustyShock.cpp`
- `DustyShock` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<DustyShock>` / `Physics_Traits<DustyShock>` specializations (`reviewed`): isothermal gas+dust shock test configuration reviewed.
- `DustDrag<DustyShock>::ComputeReciprocalStoppingTime(...)` (`reviewed`/`partial`): reciprocal stopping time `1/rho_d` model reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<DustyShock>::setInitialConditionsOnGrid(...)` (`finding`, AMR correctness): uses `Geom(0).CellSizeArray()` / `Geom(0).ProbLoArray()` (`src/problems/DustyShock/testDustyShock.cpp:70-71`) instead of `grid_elem` geometry. Refined-level IC fills would place the shock using level-0 coordinates.
- `solve_quadratic_root_in_0_1(...)` (`reviewed`): quadratic-root selection helper for analytic profile reviewed.
- `linear_interpolate(...)` (`reviewed`): bounded linear interpolation helper reviewed.
- `problem_main()` (`reviewed`/`partial`): numerical shock extraction, analytic profile integration, interpolation, error norms, and pass/fail logic reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/ParticleAccretion/testParticleAccretion.cpp`
- `AccretionProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`turnon_fextract`, `particle_in_cell_center`, `return_1_at_fail`, `sink_file`, `T0`, `mu`, `k_B`, `cs0`, `B0`, `rho0`, `t_end_over_t_b`, `M_star_in_Msun`, `uniform_density`, `refine_center`) (`reviewed`): Bondi/Bondi-Hoyle accretion test configuration and runtime knobs reviewed.
- `Particle_Traits<AccretionProblem>` / `quokka::EOS_Traits<AccretionProblem>` / `HydroSystem_Traits<AccretionProblem>` / `Physics_Traits<AccretionProblem>` specializations (`reviewed`): sink-particle + MHD + self-gravity configuration reviewed.
- `SimulationData<AccretionProblem>` specialization (`reviewed`): time/mass diagnostics storage reviewed.
- `QuokkaSimulation<AccretionProblem>::createInitialSinkParticles()` (`reviewed`/`partial`): sink particle initialization and mass/position overwrite path reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): Bondi profile interpolation table setup and gas IC fill reviewed; no confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<AccretionProblem>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): face-centered uniform magnetic field initialization reviewed.
- `QuokkaSimulation<AccretionProblem>::refineGrid(...)` (`reviewed`): central-radius tagging logic reviewed.
- `QuokkaSimulation<AccretionProblem>::computeAfterTimestep()` (`reviewed`): sink mass timeseries collection reviewed.
- `problem_main()` (`reviewed`/`partial`): parameter parsing, mass accounting, accretion-rate comparison, and optional profile plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/ParticleCreation/testParticleCreation.cpp`
- `TestParticle` (`reviewed`): empty tag type for trait specialization.
- Constants/globals (`rho0`, `dt_`, `refine_half_domain`, `B0`, particle placement constants, `SN_mass`, expected particle counts) (`reviewed`): particle-creation test setup reviewed.
- `quokka::EOS_Traits<TestParticle>` / `Particle_Traits<TestParticle>` / `HydroSystem_Traits<TestParticle>` / `Physics_Traits<TestParticle>` specializations (`reviewed`): test-particle + MHD + self-gravity configuration reviewed.
- Test enum `TestEnum` (`reviewed`): compile-time trait misuse example only.
- `QuokkaSimulation<TestParticle>::createInitialTestParticles()` (`reviewed`): ASCII particle load and integer-stage initialization reviewed.
- `quokka::ParticleCreationTraits<ParticleType::Test>::ParticleChecker<problem_t>` and ctor (`reviewed`): time-window + fixed-cell particle-creation predicate reviewed.
- `ParticleChecker::operator()(...)` (`finding`, portability): unconditionally indexes `dx[2]` when computing `k_par1/k_par2` (`src/problems/ParticleCreation/testParticleCreation.cpp:137`, `:141`) without a 3D-only guard, so the checker is not dimension-safe for 1D/2D builds.
- `quokka::ParticleCreationTraits<ParticleType::Test>::ParticleCreator<problem_t>` ctor (`reviewed`): captures particle metadata indices/runtime values reviewed.
- `ParticleCreator::operator()(...)` (`finding`, portability): unconditionally reads/writes z-components (`dx[2]`, `plo[2]`, `p.pos(2)`, `vz`) (`src/problems/ParticleCreation/testParticleCreation.cpp:183`, `:192`) without a 3D-only guard; not dimension-safe for 1D/2D builds.
- `quokka::ParticleCreationTraits<ParticleType::Test>::createParticles(...)` (`reviewed`): wrapper to generic particle-creation implementation reviewed.
- `QuokkaSimulation<TestParticle>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas + magnetic energy IC fill reviewed.
- `QuokkaSimulation<TestParticle>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): face-centered uniform magnetic field initialization reviewed.
- `problem_main()` (`reviewed`/`partial`): initialize/evolve and particle-count validation reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/ParticleRadiation/testParticleRadiation.cpp`
- `ParticleRadiationProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`mu`, `gamma_`, `rho0`, `T0`, `CV`, `initial_Erad`, `dt_`, `chat_over_c`, `formation_time`) (`reviewed`): particle-radiation test parameters reviewed (`formation_time` appears unused in this file).
- `SimulationData<ParticleRadiationProblem>` specialization (`reviewed`): particle input filename storage reviewed.
- `quokka::EOS_Traits<ParticleRadiationProblem>` / `Particle_Traits<ParticleRadiationProblem>` / `HydroSystem_Traits<ParticleRadiationProblem>` / `Physics_Traits<ParticleRadiationProblem>` / `RadSystem_Traits<ParticleRadiationProblem>` specializations (`reviewed`): stellar-pop + radiation test configuration reviewed.
- Test enum `TestEnum` (`reviewed`): compile-time trait misuse example only.
- `RadSystem<ParticleRadiationProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): constant piecewise opacity table callback reviewed.
- `QuokkaSimulation<ParticleRadiationProblem>::createInitialStochasticStellarPopParticles()` (`reviewed`): ASCII load and stage initialization of stellar-pop particles reviewed.
- `QuokkaSimulation<ParticleRadiationProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation background initialization reviewed.
- `problem_main()` (`reviewed`/`partial`): initialization, energy accounting across radiation+gas, expected table-driven emission checks, and tolerance logic reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/ParticleSF/testParticleSF.cpp`
- `ParticleSFProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`mu`, `gamma_`, `year`, `n0`, `Tamb`, `validate_initial_imf_stats`) (`reviewed`): stochastic star-formation test controls reviewed.
- `Particle_Traits<ParticleSFProblem>` / `quokka::EOS_Traits<ParticleSFProblem>` / `HydroSystem_Traits<ParticleSFProblem>` / `Physics_Traits<ParticleSFProblem>` specializations (`reviewed`): stochastic stellar-pop hydro configuration reviewed.
- `SimulationData<ParticleSFProblem>` specialization (`reviewed`): initial gas mass storage reviewed.
- `QuokkaSimulation<ParticleSFProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform Jeans-unstable gas initialization reviewed.
- `QuokkaSimulation<ParticleSFProblem>::refineGrid(...)` (`reviewed`): full-domain static refinement tagging reviewed.
- `QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()` (`finding`, robustness): `mean_mass_high_mass_stars = m_star_high_tot / n_star_high` (`src/problems/ParticleSF/testParticleSF.cpp:153`) can divide by zero if step-1 stochastic sampling produces no high-mass stars.
- `QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()` (`finding`, robustness): histogram slope diagnostic takes `std::log(hist[0])` / `std::log(hist[n_bins-1])` (`src/problems/ParticleSF/testParticleSF.cpp:187`) without checking empty bins; zero counts yield `-inf`/`nan`.
- `QuokkaSimulation<ParticleSFProblem>::computeAfterTimestep()` (`finding`, test validity): expectation checks use one-sided normalized differences without `abs(...)` (`src/problems/ParticleSF/testParticleSF.cpp:209-223`). Large underestimates can still pass because negative relative errors satisfy `< tol`.
- `problem_main()` (`finding`, test validity/robustness): final mass check also uses one-sided relative difference and divides by `m_star_tot2` (`src/problems/ParticleSF/testParticleSF.cpp:325-327`) without `abs(...)` or a zero guard, weakening/falsifying failure detection when stellar mass is small or underpredicted.
- `problem_main()` (`reviewed`): initialization, RNG seeding, restart low-mass-cap validation, and end-of-run mass check orchestration reviewed.

### `src/problems/ParticleSink/testParticleSink.cpp`
- `SinkProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`refine_half_domain`, `mu`, `gamma_`, `rho0`, `T0`, `CV`, `year`, `dt_init`, `B0`, `particles_file`) (`reviewed`): sink-particle test configuration reviewed.
- `Particle_Traits<SinkProblem>` / `quokka::EOS_Traits<SinkProblem>` / `HydroSystem_Traits<SinkProblem>` / `Physics_Traits<SinkProblem>` specializations (`reviewed`): sink + MHD + self-gravity configuration reviewed.
- `SimulationData<SinkProblem>` specialization (`reviewed`): boost-velocity storage reviewed.
- `QuokkaSimulation<SinkProblem>::createInitialSinkParticles()` (`reviewed`): sink particle load + boost-velocity application reviewed.
- `QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): boosted uniform gas + magnetic energy IC fill reviewed.
- `QuokkaSimulation<SinkProblem>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): face-centered uniform magnetic field initialization reviewed.
- `QuokkaSimulation<SinkProblem>::refineGrid(...)` (`finding`, AMR region selection): normalized coordinates are computed as `((i+0.5)*dx)/(phi-plo)` (`src/problems/ParticleSink/testParticleSink.cpp:152-154`) without subtracting `plo`, so the selected refinement subregion shifts if the domain lower bound is nonzero.
- `problem_main()` (`finding`, correctness): `AMREX_ASSERT_WITH_MESSAGE(boost_vel_x != NAN, ...)` (`src/problems/ParticleSink/testParticleSink.cpp:167-169`) is ineffective because comparisons with `NaN` are always true. Missing `boost_vel_x` input is not reliably detected.
- `problem_main()` (`reviewed`): three-phase mass conservation / analytic / Galilean-invariance checks and plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp`
- `SinkProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`M_sol`, `mu`, `gamma_`, `rho0`, `T0`, `CV`, `year`, `cs`, `B0`) (`reviewed`): sink-formation test setup constants reviewed (`M_sol` appears unused in this file).
- `Particle_Traits<SinkProblem>` / `quokka::EOS_Traits<SinkProblem>` / `HydroSystem_Traits<SinkProblem>` / `Physics_Traits<SinkProblem>` specializations (`reviewed`): sink + MHD + self-gravity configuration reviewed.
- `QuokkaSimulation<SinkProblem>::setInitialConditionsOnGrid(...)` (`finding`, AMR correctness): uses `geom[0].ProbLoArray()` / `geom[0].CellSizeArray()` (`src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp:70-71`) instead of `grid_elem` geometry. Refined-level IC fills would evaluate Jeans threshold and peak-cell placement with level-0 spacing/coordinates.
- `QuokkaSimulation<SinkProblem>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): face-centered uniform magnetic field initialization reviewed.
- `QuokkaSimulation<SinkProblem>::refineGrid(...)` (`reviewed`): full-domain static refinement tagging reviewed.
- `problem_main()` (`reviewed`): formation-step particle-count and mass-conservation checks, continued accretion run, and diagnostic plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/RadBeam/testRadBeam.cpp`
- `BeamProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`kappa0`, `rho0`, `T_hohlraum`, `T_initial`, `a_rad`, `c`) (`reviewed`): streaming-beam test constants reviewed.
- `quokka::EOS_Traits<BeamProblem>` / `RadSystem_Traits<BeamProblem>` / `Physics_Traits<BeamProblem>` specializations (`reviewed`): radiation streaming test configuration reviewed.
- `RadSystem<BeamProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): zero-opacity closures reviewed.
- `AMRSimulation<BeamProblem>::setCustomBoundaryConditions(...)` (`finding`, portability): only defines index unpacking for `AMREX_SPACEDIM == 2` or `3` (`src/problems/RadBeam/testRadBeam.cpp:76-82`). A 1D build leaves `i/j/k` undefined and this specialization is not dimension-safe.
- `AMRSimulation<BeamProblem>::setCustomBoundaryConditions(...)` (`reviewed`): inflow/reflecting radiation boundary logic and gas extrapolation reviewed; no additional confirmed bug in inspected implementation this pass.
- `QuokkaSimulation<BeamProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation initialization reviewed.
- `QuokkaSimulation<BeamProblem>::refineGrid(...)` (`finding`, portability): refinement indicator unconditionally accesses y-neighbors (`state(i, j±1, ...)`) (`src/problems/RadBeam/testRadBeam.cpp:257-258`), so the implementation is hard-coded for 2D+ and not 1D-safe.
- `QuokkaSimulation<BeamProblem>::refineGrid(...)` (`reviewed`): radiation-gradient tagging logic reviewed.
- `problem_main()` (`reviewed`): BC setup, runtime configuration, initialize/evolve path reviewed (no pass/fail assertion in this driver).

### `src/problems/RadDust/testRadDust.cpp`
- `DustProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`beta_order_`, `c`, `chat`, `v0`, `chi0`, `T0`, `rho0`, `a_rad`, `mu`, `k_B`, `max_time`, `Erad0`, `erad_floor`) (`reviewed`): single-group dust-radiation coupling test parameters reviewed.
- `SimulationData<DustProblem>` specialization (`reviewed`): temperature/radiation timeseries storage reviewed.
- `quokka::EOS_Traits<DustProblem>` / `RadSystem_Traits<DustProblem>` / `ISM_Traits<DustProblem>` / `Physics_Traits<DustProblem>` specializations (`reviewed`): single-group gas-radiation coupling configuration reviewed.
- `RadSystem<DustProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): opacity closures reviewed.
- `RadSystem<DustProblem>::ComputeThermalRadiationSingleGroup(...)` / `ComputeThermalRadiationTempDerivativeSingleGroup(...)` (`reviewed`): linearized thermal-emission closures reviewed.
- `QuokkaSimulation<DustProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation IC fill reviewed.
- `QuokkaSimulation<DustProblem>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction and temperature reconstruction reviewed.
- `problem_main()` (`reviewed`/`partial`): evolve, CSV exact-solution load, interpolation/error computation, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadDustMG/testRadDustMG.cpp`
- `DustProblem` (`reviewed`): empty tag type for trait specialization (multigroup version).
- Constants (`beta_order_`, `c`, `chat`, `v0`, `chi0`, `T0`, `rho0`, `a_rad`, `mu`, `k_B`, `max_time`, `Erad0`, `erad_floor`) (`reviewed`): multigroup dust-radiation coupling test parameters reviewed.
- `SimulationData<DustProblem>` specialization (`reviewed`): temperature/radiation timeseries storage reviewed.
- `quokka::EOS_Traits<DustProblem>` / `Physics_Traits<DustProblem>` / `RadSystem_Traits<DustProblem>` / `ISM_Traits<DustProblem>` specializations (`reviewed`): multigroup gas-radiation coupling configuration reviewed.
- `RadSystem<DustProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): piecewise-opacity callback reviewed.
- `RadSystem<DustProblem>::ComputeThermalRadiationMultiGroup(...)` / `ComputeThermalRadiationTempDerivativeMultiGroup(...)` (`reviewed`): linearized multigroup thermal-emission closures reviewed.
- `QuokkaSimulation<DustProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup uniform gas+radiation IC fill reviewed.
- `QuokkaSimulation<DustProblem>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction and aggregate radiation-energy tracking reviewed.
- `problem_main()` (`reviewed`/`partial`): evolve, exact-data load/interpolation, error computation, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadForce/testRadForce.cpp`
- `TubeProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`kappa0`, `mu`, `gamma_gas`, `a0`, `tau`, `rho0`, `Mach0`, `Mach1`, `Frad0`, `g0`, `Lx`) (`reviewed`): radiation-force tube test parameters reviewed.
- `quokka::EOS_Traits<TubeProblem>` / `Physics_Traits<TubeProblem>` / `RadSystem_Traits<TubeProblem>` specializations (`reviewed`): radiation-force hydrodynamic tube configuration reviewed.
- `RadSystem<TubeProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): opacity closures reviewed.
- `QuokkaSimulation<TubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform radiation+gas background initialization reviewed.
- `AMRSimulation<TubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): constant inflow Dirichlet lower-x boundary helper usage reviewed.
- `problem_main()` (`finding`, robustness): plotting stride uses `int s = nx / 64` (`src/problems/RadForce/testRadForce.cpp:298`) and passes it to `strided_vector_from(...)` (`:308`) without `s >= 1` guard. For `nx < 64` and `HAVE_PYTHON`, this yields `stride == 0`, which is unsafe for `strided_vector_from()`.
- `problem_main()` (`reviewed`): configuration, evolve, exact-solution file load, interpolation/error norm, and plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/RadLineCooling/testRadLineCooling.cpp`
- `CoolingProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`export_csv`, `cooling_rate`, `CR_heating_rate`, `c`, `chat`, `v0`, `kappa0`, `T0`, `rho0`, `a_rad`, `mu`, `C_V`, `nu_unit`, `erad_floor`, `max_time`) (`reviewed`): single-group line-cooling test parameters reviewed.
- `SimulationData<CoolingProblem>` specialization (`reviewed`): gas-temperature and radiation-energy timeseries storage reviewed.
- `quokka::EOS_Traits<CoolingProblem>` / `Physics_Traits<CoolingProblem>` / `RadSystem_Traits<CoolingProblem>` / `ISM_Traits<CoolingProblem>` specializations (`reviewed`): line-cooling + CR-heating test configuration reviewed.
- `RadSystem<CoolingProblem>::DefineNetCoolingRate(...)` / `DefineNetCoolingRateTempDerivative(...)` / `DefineCosmicRayHeatingRate(...)` (`reviewed`): linear cooling/heating closures reviewed.
- `RadSystem<CoolingProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` / `DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): zero-opacity callbacks reviewed.
- `QuokkaSimulation<CoolingProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation IC fill reviewed.
- `QuokkaSimulation<CoolingProblem>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction and gas/rad state reconstruction reviewed.
- `problem_main()` (`reviewed`/`partial`): exact-solution construction, interpolation-free error check, CSV export, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadLineCoolingMG/testRadLineCoolingMG.cpp`
- `CoolingProblemMG` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`export_csv`, `n_groups_`, `rad_boundaries_`, `chat_over_c`, `v0`, `kappa0`, `T0`, `rho0`, `a_rad`, `mu`, `C_V`, `nu_unit`, `Erad_bar`, `Erad_floor_`, `Erad_FUV`, `max_time`, `line_index`, `cooling_rate`, `CR_heating_rate`, `PE_rate`) (`reviewed`): multigroup line-cooling/PE-heating test parameters reviewed.
- `SimulationData<CoolingProblemMG>` specialization (`reviewed`): gas-temperature and line-group radiation-energy timeseries storage reviewed.
- `quokka::EOS_Traits<CoolingProblemMG>` / `Physics_Traits<CoolingProblemMG>` / `RadSystem_Traits<CoolingProblemMG>` / `ISM_Traits<CoolingProblemMG>` specializations (`reviewed`): multigroup line-cooling + PE-heating configuration reviewed.
- `RadSystem<CoolingProblemMG>::DefinePhotoelectricHeatingE1Derivative(...)` / `DefineNetCoolingRate(...)` / `DefineNetCoolingRateTempDerivative(...)` / `DefineCosmicRayHeatingRate(...)` (`reviewed`): multigroup heating/cooling closures reviewed.
- `RadSystem<CoolingProblemMG>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): zero-opacity callback reviewed.
- `QuokkaSimulation<CoolingProblemMG>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup uniform gas+radiation IC fill reviewed.
- `QuokkaSimulation<CoolingProblemMG>::computeAfterTimestep()` (`reviewed`/`partial`): slice-based timeseries extraction and line-group radiation tracking reviewed.
- `problem_main()` (`reviewed`/`partial`): exact-solution construction/interpolation, error check, CSV export, coupling-mode plotting, and pass/fail threshold reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadMarshak/testRadMarshak.cpp`
- `SuOlsonProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`eps_SuOlson`, `kappa`, `rho0`, `T_hohlraum`, `a_rad`, `c`, `alpha_SuOlson`, `T_initial`) (`reviewed`): dimensionless Su-Olson setup parameters reviewed.
- `quokka::EOS_Traits<SuOlsonProblem>` / `RadSystem_Traits<SuOlsonProblem>` / `Physics_Traits<SuOlsonProblem>` specializations (`reviewed`): single-group radiation-only test configuration reviewed.
- `RadSystem<SuOlsonProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `quokka::EOS<SuOlsonProblem>::ComputeTgasFromEint(...)` / `ComputeEintFromTgas(...)` / `ComputeEintTempDerivative(...)` (`reviewed`): Su-Olson analytic EOS specializations reviewed.
- `AMRSimulation<SuOlsonProblem>::setCustomBoundaryConditions(...)` (`reviewed`): Marshak left boundary + constant right boundary and gas ghost fill reviewed.
- `QuokkaSimulation<SuOlsonProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas/radiation IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): setup, evolve, exact-data file load/interpolation, L1 error calculation, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadMarshakCGS/testRadMarshakCGS.cpp`
- `SuOlsonProblemCgs` (`reviewed`): empty tag type for trait specialization.
- Constants (`eps_SuOlson`, `kappa`, `rho0`, `T_hohlraum`, `a_rad`, `alpha_SuOlson`, `T_initial`) (`reviewed`): CGS Su-Olson setup parameters reviewed.
- `quokka::EOS_Traits<SuOlsonProblemCgs>` / `RadSystem_Traits<SuOlsonProblemCgs>` / `Physics_Traits<SuOlsonProblemCgs>` specializations (`reviewed`): single-group CGS radiation-only configuration reviewed.
- `RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `quokka::EOS<SuOlsonProblemCgs>::ComputeTgasFromEint(...)` / `ComputeEintFromTgas(...)` / `ComputeEintTempDerivative(...)` (`reviewed`): Su-Olson analytic EOS specializations reviewed.
- `AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(...)` (`reviewed`): Marshak left boundary + constant right boundary and gas ghost fill reviewed.
- `QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(...)` (`finding`, code quality/correctness hygiene): `state_cc(..., radEnergy_index)` is assigned twice consecutively (`src/problems/RadMarshakCGS/testRadMarshakCGS.cpp:183-184`). The duplicate write is likely harmless but indicates a copy/paste bug in the IC kernel.
- `problem_main()` (`reviewed`/`partial`): setup, evolve, exact-data load/interpolation, L1 error check, and optional plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/RadMarshakAsymptotic/testRadMarshakAsymptotic.cpp`
- `SuOlsonProblemCgs` (`reviewed`): empty tag type for trait specialization.
- Constants (`kappa`, `rho0`, `T_hohlraum`, `T_initial`, `a_rad`, `Erad_floor_`) (`reviewed`): asymptotic-diffusion Marshak setup parameters reviewed.
- `quokka::EOS_Traits<SuOlsonProblemCgs>` / `RadSystem_Traits<SuOlsonProblemCgs>` / `Physics_Traits<SuOlsonProblemCgs>` specializations (`reviewed`): asymptotic-preserving radiation-only configuration reviewed.
- `RadSystem<SuOlsonProblemCgs>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` / `ComputeEddingtonFactor(...)` (`reviewed`): temperature-dependent opacity and fixed Eddington-factor closures reviewed.
- `AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(...)` (`reviewed`): first-order Marshak boundary implementation and ghost fill logic reviewed.
- `QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas/radiation IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): asymptotic-preserving test setup, runtime toggle parsing, evolve, exact-data interpolation, error metric, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadMarshakDust/testRadMarshakDust.cpp`
- `MarshakProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`kappa1`, `kappa2`, `c`, `c_hat_over_c_`, `c_hat`, `rho0`, `CV`, `mu`, `initial_T`, `a_rad`, `erad_floor`, `initial_Trad`, `T_rad_L`, `EradL`, `T_end_exact`, `n_group_`, `radBoundaries_`, `opacity_model_`) (`reviewed`): multigroup dust Marshak test parameters reviewed.
- `quokka::EOS_Traits<MarshakProblem>` / `Physics_Traits<MarshakProblem>` / `RadSystem_Traits<MarshakProblem>` / `ISM_Traits<MarshakProblem>` specializations (`reviewed`): multigroup dust/gas thermal-coupling configuration reviewed.
- `RadSystem<MarshakProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `RadSystem<MarshakProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): piecewise-constant multigroup opacity callback reviewed.
- `QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup gas/radiation IC fill reviewed.
- `AMRSimulation<MarshakProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left Dirichlet multigroup radiation inflow + gas ghost fill via helper reviewed.
- `problem_main()` (`reviewed`/`partial`): runtime opacity parsing, evolve, analytic comparison, error norm computation, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp`
- `MarshakProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/constants (`PE_rate`, `kappa1`, `kappa2`, `dust_on`, `PE_on`, `gas_dust_coupling_threshold_`, `c`, `c_hat_over_c_`, `c_hat_`, `rho0`, `CV`, `mu`, `initial_T`, `a_rad`, `erad_floor`, `T_rad_L`, `EradL`, `n_group_`, `radBoundaries_`, `opacity_model_`) (`reviewed`): dust + PE-heating Marshak test parameters reviewed.
- `quokka::EOS_Traits<MarshakProblem>` / `Physics_Traits<MarshakProblem>` / `RadSystem_Traits<MarshakProblem>` / `ISM_Traits<MarshakProblem>` specializations (`reviewed`): dust/PE coupling configuration reviewed.
- `RadSystem<MarshakProblem>::DefinePhotoelectricHeatingE1Derivative(...)` (`reviewed`): constant PE-heating derivative callback reviewed.
- `RadSystem<MarshakProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): piecewise-constant multigroup opacity callback reviewed.
- `QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup gas/radiation IC fill reviewed.
- `AMRSimulation<MarshakProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left Dirichlet multigroup radiation inflow + gas ghost fill via helper reviewed.
- `problem_main()` (`reviewed`/`partial`): runtime opacity parsing, coupling-mode branch, evolve, analytic comparison, error norm computation, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadMarshakVaytet/testRadMarshakVaytet.cpp`
- Constants/config (`n_groups_`, `opacity_model_`, `kappa0`, `nu_pivot`, `n_coll`, `the_model`, `group_edges_`, `group_opacities_`) (`reviewed`): multigroup Vaytet-style variable-opacity configuration and compile-time group-bin tables reviewed.
- `SuOlsonProblemCgs` (`reviewed`): empty tag type for trait specialization.
- Constants (`max_step_`, `rho0`, `T_initial`, `T_L`, `T_R`, `rho_C_V`, `c_v`, `mu`, `a_rad`, `Erad_floor_`) (`reviewed`): CGS Marshak/Vaytet setup parameters reviewed.
- `quokka::EOS_Traits<SuOlsonProblemCgs>` / `Physics_Traits<SuOlsonProblemCgs>` / `RadSystem_Traits<SuOlsonProblemCgs>` specializations (`reviewed`): multigroup radiation-only configuration reviewed.
- `RadSystem<SuOlsonProblemCgs>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): model-dependent multigroup opacity callback reviewed (constant/nu-dependent/T-dependent/PPL branches).
- `AMRSimulation<SuOlsonProblemCgs>::setCustomBoundaryConditions(...)` (`reviewed`): left/right constant Dirichlet multigroup boundary ghost fill via helpers reviewed.
- `QuokkaSimulation<SuOlsonProblemCgs>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup thermal-radiation IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): setup, evolve, multigroup-to-collapsed diagnostics/CSV export, and plotting reviewed; this driver currently hardcodes `const int status = 0` (`src/problems/RadMarshakVaytet/testRadMarshakVaytet.cpp:295`) and leaves error checks commented out, so it functions as a data/plot generator rather than an automated pass/fail test.

### `src/problems/RadStreaming/testRadStreaming.cpp`
- `StreamingProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`initial_Erad`, `initial_Egas`, `c`, `chat`, `kappa0`, `rho`) (`reviewed`): free-streaming test parameters reviewed.
- `quokka::EOS_Traits<StreamingProblem>` / `Physics_Traits<StreamingProblem>` / `RadSystem_Traits<StreamingProblem>` specializations (`reviewed`): radiation-only streaming configuration reviewed.
- `RadSystem<StreamingProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas + (possibly multigroup) radiation IC fill reviewed.
- `AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left streaming inflow and right constant/outflow Dirichlet ghost fill via helper functions reviewed.
- `problem_main()` (`reviewed`): setup, evolve, analytic top-hat streaming comparison, L1 error metric, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadStreamingY/testRadStreamingY.cpp`
- `StreamingProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`initial_Erad`, `initial_Egas`, `c`, `chat`, `kappa0`, `rho`) (`reviewed`): free-streaming-in-y test parameters reviewed.
- `quokka::EOS_Traits<StreamingProblem>` / `Physics_Traits<StreamingProblem>` / `RadSystem_Traits<StreamingProblem>` specializations (`reviewed`): radiation-only streaming configuration reviewed.
- `RadSystem<StreamingProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `QuokkaSimulation<StreamingProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation IC fill reviewed.
- `AMRSimulation<StreamingProblem>::setCustomBoundaryConditions(...)` (`finding`, portability): applies `setConstantDirichletBCLo<1>` / `setConstantDirichletBCHi<1>` unconditionally (`src/problems/RadStreamingY/testRadStreamingY.cpp:139-140`), so the specialization is not 1D-safe.
- `problem_main()` (`finding`, portability): unconditionally configures `BCs_cc[*].setLo/Hi(1, ...)` and extracts a slice with `fextract(..., 1, 0.0)` (`src/problems/RadStreamingY/testRadStreamingY.cpp:160-161`, `:185`), which is invalid for 1D builds without a dimension guard.
- `problem_main()` (`reviewed`): y-streaming analytic comparison, L1 error metric, and optional plotting reviewed.

### `src/problems/RadShadow/testRadShadow.cpp`
- `ShadowProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`sigma0`, `rho_bg`, `rho_clump`, `T_hohlraum`, `T_initial`, `a_rad`, `c`) (`reviewed`): 2D radiation-shadow test parameters reviewed.
- `quokka::EOS_Traits<ShadowProblem>` / `RadSystem_Traits<ShadowProblem>` / `Physics_Traits<ShadowProblem>` specializations (`reviewed`): radiation-only shadowing configuration reviewed.
- `RadSystem<ShadowProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): density-dependent opacity closures reviewed.
- `AMRSimulation<ShadowProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left free-streaming radiation inflow + gas outflow extrapolation ghost fill reviewed.
- `QuokkaSimulation<ShadowProblem>::setInitialConditionsOnGrid(...)` (`finding`, portability): clump IC construction unconditionally uses `prob_lo[1]` / `dx[1]` (`src/problems/RadShadow/testRadShadow.cpp:127`), so the implementation is not 1D-safe without a dimension guard.
- `QuokkaSimulation<ShadowProblem>::refineGrid(...)` (`finding`, portability): refinement indicator unconditionally samples y-neighbors (`state(i, j±1, ...)`) (`src/problems/RadShadow/testRadShadow.cpp:168-169`), making this tagging logic hard-coded for 2D+.
- `problem_main()` (`reviewed`): 2D setup, custom BC construction, stiffness diagnostic print, evolve path reviewed (driver is primarily runtime/visual output, no automated pass/fail check).

### `src/problems/RadTophat/testRadTophat.cpp`
- `TophatProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`kelvin_to_eV`, `kappa_wall`, `rho_wall`, `kappa_pipe`, `rho_pipe`, `T_hohlraum`, `T_initial`, `c_v`, `a_rad`, `c`) (`reviewed`): Gentile tophat test parameters reviewed.
- `quokka::EOS_Traits<TophatProblem>` / `RadSystem_Traits<TophatProblem>` / `Physics_Traits<TophatProblem>` specializations (`reviewed`): radiation-only diffusion test configuration reviewed.
- `RadSystem<TophatProblem>::ComputePlanckOpacity(...)` (`finding`, correctness): the fallback branch uses `AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true, "opacity not defined!")` (`src/problems/RadTophat/testRadTophat.cpp:80`), which never fails; unsupported densities silently continue with `kappa == 0`.
- `RadSystem<TophatProblem>::ComputeFluxMeanOpacity(...)` (`reviewed`): delegates to Planck opacity closure.
- `quokka::EOS<TophatProblem>::ComputeTgasFromEint(...)` / `ComputeEintFromTgas(...)` / `ComputeEintTempDerivative(...)` (`reviewed`): constant-`c_v` EOS specializations reviewed.
- `RadSystem<TophatProblem>::ComputeEddingtonFactor(...)` (`reviewed`): Minerbo-style closure approximation reviewed.
- `AMRSimulation<TophatProblem>::setCustomBoundaryConditions(...)` (`finding`, portability): index unpacking is only defined for `AMREX_SPACEDIM == 2/3` (`src/problems/RadTophat/testRadTophat.cpp:136-142`), but the function unconditionally uses `j` and `prob_lo[1]` (`:150`), so a 1D build is not dimension-safe.
- `QuokkaSimulation<TophatProblem>::setInitialConditionsOnGrid(...)` (`finding`, portability): tophat geometry classification unconditionally uses y-coordinates (`src/problems/RadTophat/testRadTophat.cpp:214`) and is not 1D-safe without a dimension guard.
- `problem_main()` (`reviewed`): runtime/BC setup and evolve path reviewed (no automated accuracy check in this driver).

### `src/problems/RadMatterCoupling/testRadMatterCoupling.cpp`
- `CouplingProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`eps_SuOlson`, `a_rad`, `alpha_SuOlson`, `Erad0`, `Egas0`, `rho0`) (`reviewed`): matter-radiation coupling test parameters reviewed.
- `SimulationData<CouplingProblem>` specialization (`reviewed`): time/temperature history vectors reviewed.
- `quokka::EOS_Traits<CouplingProblem>` / `RadSystem_Traits<CouplingProblem>` / `Physics_Traits<CouplingProblem>` specializations (`reviewed`): radiation-only coupling configuration reviewed.
- `RadSystem<CouplingProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): unit-opacity closures reviewed.
- `quokka::EOS<CouplingProblem>::ComputeTgasFromEint(...)` / `ComputeEintFromTgas(...)` / `ComputeEintTempDerivative(...)` (`reviewed`): Su-Olson-style analytic EOS specializations reviewed.
- `QuokkaSimulation<CouplingProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation IC fill reviewed.
- `QuokkaSimulation<CouplingProblem>::computeAfterTimestep()` (`reviewed`): slice extraction and temperature-history accumulation reviewed.
- `problem_main()` (`reviewed`/`partial`): evolve path, analytic solution construction/interpolation, L1 error metric, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadMatterCouplingRSLA/testRadMatterCouplingRSLA.cpp`
- `CouplingProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`chat_over_c`, `c_rsla`, `eps_SuOlson`, `a_rad`, `alpha_SuOlson`, `Erad0`, `Egas0`, `rho0`) (`reviewed`): RSLA coupling test parameters reviewed.
- `SimulationData<CouplingProblem>` specialization (`reviewed`): time/temperature history vectors reviewed.
- `quokka::EOS_Traits<CouplingProblem>` / `RadSystem_Traits<CouplingProblem>` / `Physics_Traits<CouplingProblem>` specializations (`reviewed`): RSLA radiation-only coupling configuration reviewed.
- `RadSystem<CouplingProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): unit-opacity closures reviewed.
- `quokka::EOS<CouplingProblem>::ComputeTgasFromEint(...)` / `ComputeEintFromTgas(...)` / `ComputeEintTempDerivative(...)` (`reviewed`): Su-Olson-style analytic EOS specializations reviewed.
- `QuokkaSimulation<CouplingProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform gas+radiation IC fill reviewed.
- `QuokkaSimulation<CouplingProblem>::computeAfterTimestep()` (`reviewed`): slice extraction and temperature-history accumulation reviewed.
- `problem_main()` (`reviewed`/`partial`): RSLA/no-RSLA analytic trajectories, L1 error metric, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadSuOlson/testRadSuOlson.cpp`
- `MarshakProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`eps_SuOlson`, `kappa`, `rho0`, `T_hohlraum`, `x0`, `t0`, `a_rad`, `c`, `alpha_SuOlson`, `Q`, `S`, `initial_Egas`, `initial_Erad`) (`reviewed`): Su-Olson source-driven diffusion test parameters reviewed.
- `RadSystem_Traits<MarshakProblem>` / `quokka::EOS_Traits<MarshakProblem>` / `Physics_Traits<MarshakProblem>` specializations (`reviewed`): radiation-only source-term test configuration reviewed.
- `RadSystem<MarshakProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): `kappa/rho` closures reviewed.
- `quokka::EOS<MarshakProblem>::ComputeTgasFromEint(...)` / `ComputeEintFromTgas(...)` / `ComputeEintTempDerivative(...)` (`reviewed`): Su-Olson analytic EOS specializations reviewed.
- `RadSystem<MarshakProblem>::SetRadEnergySource(...)` (`finding`, domain-origin correctness): source-cell overlap coordinates are computed as `xl = i*dx` / `xr = (i+1)*dx` (`src/problems/RadSuOlson/testRadSuOlson.cpp:138-139`) while the `prob_lo` argument is ignored (`:134`). If the domain lower bound is nonzero, the source region `[0, x0]` is shifted incorrectly.
- `QuokkaSimulation<MarshakProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform low-energy gas+radiation IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): source-driven evolve path, tabulated exact comparisons, interpolation, L1 error metric, and plotting reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/RadTube/testRadTube.cpp`
- `TubeProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`kappa0`, `mu`, `gamma_gas`, `rho0`, `T_lo`, `rho1`, `T_hi`, `a_rad`, `a0`) (`reviewed`): radiation-pressure tube parameters reviewed.
- `quokka::EOS_Traits<TubeProblem>` / `Physics_Traits<TubeProblem>` / `RadSystem_Traits<TubeProblem>` specializations (`reviewed`): hydro+radiation multigroup tube configuration reviewed.
- `RadSystem<TubeProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): constant piecewise-opacity callback reviewed.
- `SimulationData<TubeProblem>` specialization (`reviewed`): device vectors for tabulated ICs reviewed.
- `QuokkaSimulation<TubeProblem>::preCalculateInitialConditions()` (`reviewed`): tabulated IC file load and host-to-device copy path reviewed.
- `QuokkaSimulation<TubeProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): interpolated hydro+radiation IC fill and multigroup Planck partitioning reviewed.
- `AMRSimulation<TubeProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left/right Dirichlet hydro+radiation ghost-fill construction reviewed.
- `problem_main()` (`reviewed`/`partial`): setup, evolve, initial/final extraction, exact-reference interpolation, L1 error metric, and optional plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroShock/testRadhydroShock.cpp`
- `ShockProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`a_rad`, `sigma_a`, `Mach0`, `c_s0`, `c`, `k_B`, `kappa`, `gamma_gas`, `mu`, `c_v`, `T_lo`, `rho0`, `v0`, `T_hi`, `rho1`, `v1`, `chat`, `Ggrav`, `Erad0`, `Egas0`, `Erad1`, `Egas1`, `shock_position`) (`reviewed`): dimensionless radiative-shock setup parameters reviewed.
- `RadSystem_Traits<ShockProblem>` / `quokka::EOS_Traits<ShockProblem>` / `Physics_Traits<ShockProblem>` specializations (`reviewed`): hydro+radiation shock configuration reviewed.
- `RadSystem<ShockProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` / `ComputeEddingtonFactor(...)` (`reviewed`): opacity and Eddington closures reviewed.
- `AMRSimulation<ShockProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left/right shock-state Dirichlet ghost fill construction reviewed.
- `QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): discontinuous shock-state IC fill and scalar zeroing reviewed.
- `problem_main()` (`reviewed`/`partial`): runtime setup, evolve, exact-solution file load/interpolation, L1 temperature error metric, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroShockCGS/testRadhydroShockCGS.cpp`
- `ShockProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`a_rad`, `c`, `k_B`, `c_s0`, `kappa`, `gamma_gas`, `c_v`, `T_low`, `rho0`, `v0`, `T_hi`, `rho1`, `v1`, `chat`, `Erad0`, `Egas0`, `Erad1`, `Egas1`, `shock_position`, `Lx`) (`reviewed`): CGS radiative-shock setup parameters reviewed.
- `RadSystem_Traits<ShockProblem>` / `quokka::EOS_Traits<ShockProblem>` / `Physics_Traits<ShockProblem>` specializations (`reviewed`): hydro+radiation shock configuration with custom-unit-system test reviewed.
- `RadSystem<ShockProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` / `ComputeEddingtonFactor(...)` (`reviewed`): opacity and Eddington closures reviewed.
- `AMRSimulation<ShockProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left/right shock-state Dirichlet ghost fill construction reviewed.
- `QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): discontinuous shock-state IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): runtime setup, evolve, optional exact-solution file compare, L1 error metric, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroShockMultigroup/testRadhydroShockMultigroup.cpp`
- `ShockProblem` (`reviewed`): empty tag type for trait specialization.
- Constants (`a_rad`, `c`, `k_B`, `c_s0`, `kappa`, `gamma_gas`, `c_v`, `T_lo`, `rho0`, `v0`, `T_hi`, `rho1`, `v1`, `chat`, `Erad0`, `Erad_floor_`, `Egas0`, `Egas1`, `shock_position`, `Lx`) (`reviewed`): multigroup radiative-shock setup parameters reviewed.
- `Physics_Traits<ShockProblem>` / `RadSystem_Traits<ShockProblem>` / `quokka::EOS_Traits<ShockProblem>` specializations (`reviewed`): multigroup hydro+radiation shock configuration reviewed.
- `RadSystem<ShockProblem>::DefineOpacityExponentsAndLowerValues(...)` / `ComputeEddingtonFactor(...)` (`reviewed`): grey-per-group opacity callback and Eddington closure reviewed.
- `AMRSimulation<ShockProblem>::setCustomBoundaryConditions(...)` (`reviewed`): left/right multigroup shock-state Dirichlet ghost fill construction reviewed.
- `QuokkaSimulation<ShockProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup discontinuous shock-state IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): runtime setup, evolve, optional exact-solution file compare, L1 error metric, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroShell/testRadhydroShell.cpp`
- `ShellProblem` (`reviewed`): empty tag type for trait specialization.
- Constants/config (`simulate_full_box`, `a_rad`, `c`, `a0`, `chat`, `k_B`, `m_H`, `gamma_gas`, `Msun`, `parsec_in_cm`, `specific_luminosity`, `GMC_mass`, `epsilon`, `M_shell`, `L_star`, `r_0`, `sigma_star`, `H_shell`, `kappa0`, `rho_0`, `c_v`) (`reviewed`): radiation-pressure shell setup parameters reviewed.
- `quokka::EOS_Traits<ShellProblem>` / `RadSystem_Traits<ShellProblem>` / `HydroSystem_Traits<ShellProblem>` / `Physics_Traits<ShellProblem>` specializations (`reviewed`): 3D radhydro shell configuration reviewed.
- `RadSystem<ShellProblem>::SetRadEnergySource(...)` (`finding`, portability): source kernel unconditionally uses z-dimension geometry (`prob_lo[2]`, `dx[2]`, `prob_hi[2]`) (`src/problems/RadhydroShell/testRadhydroShell.cpp:102`, `:114`) without compile-time dimension guards. `problem_main()` is 3D-gated, but this specialization itself is not dimension-safe for 1D/2D builds.
- `RadSystem<ShellProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed (`ComputePlanckOpacity` contains a harmless extra `;`).
- `SimulationData<ShellProblem>` specialization (`reviewed`): radial-profile host/device tables reviewed.
- `QuokkaSimulation<ShellProblem>::preCalculateInitialConditions()` (`reviewed`): table-file parsing and host-to-device copy path reviewed.
- `QuokkaSimulation<ShellProblem>::setInitialConditionsOnGrid(...)` (`finding`, portability): IC kernel and source-center setup unconditionally use z-coordinate geometry (`src/problems/RadhydroShell/testRadhydroShell.cpp:196`, `:212`), so the specialization is not dimension-safe outside 3D.
- `QuokkaSimulation<ShellProblem>::refineGrid(...)` (`finding`, portability): refinement indicator unconditionally samples z-neighbors (`state(i,j,k±1,...)`) (`src/problems/RadhydroShell/testRadhydroShell.cpp:277-278`), hard-coding 3D behavior in an unguarded specialization.
- `problem_main()` (`reviewed`): explicit 3D-only driver setup, octant/full-box BC configuration, evolve path reviewed.

### `src/problems/RadhydroUniformAdvecting/testRadhydroUniformAdvecting.cpp`
- `PulseProblem` (`reviewed`): empty tag type for trait specialization.
- Constants/config (`c`, `beta_order_`, `v0`, `kappa0`, `chat_over_c`, `T0`, `rho0`, `a_rad`, `mu`, `k_B`, `max_time`, `Erad0`, `Erad_beta2`) (`reviewed`): uniform advecting-radiation test parameters reviewed.
- `quokka::EOS_Traits<PulseProblem>` / `RadSystem_Traits<PulseProblem>` / `Physics_Traits<PulseProblem>` specializations (`reviewed`): hydro+radiation uniform-advection configuration reviewed.
- `RadSystem<PulseProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity closures reviewed.
- `QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): beta-order-dependent equilibrium IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): runtime setup, evolve, exact uniform-state reconstruction, L1 error metric, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroPulse/testRadhydroPulse.cpp`
- `PulseProblem` / `AdvPulseProblem` (`reviewed`): empty tag types for non-advecting/advecting pulse variants.
- Constants/config (`beta_order_`, `T_low`, `T_hi`, `rho0`, `a_rad`, `c`, `chat`, `width`, `erad_floor`, `mu`, `k_B`, `kappa0`, `v0_adv`) (`reviewed`): static-diffusion pulse test parameters reviewed.
- `quokka::EOS_Traits<PulseProblem>` / `quokka::EOS_Traits<AdvPulseProblem>` / `RadSystem_Traits<PulseProblem>` / `RadSystem_Traits<AdvPulseProblem>` / `Physics_Traits<PulseProblem>` / `Physics_Traits<AdvPulseProblem>` specializations (`reviewed`): hydro+radiation pulse configurations reviewed.
- `compute_initial_Tgas(...)` / `compute_exact_rho(...)` (`reviewed`): Gaussian pulse temperature and equilibrium density profiles reviewed.
- `RadSystem<PulseProblem>::ComputePlanckOpacity(...)` / `RadSystem<AdvPulseProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` overloads (`reviewed`): constant-opacity closures reviewed.
- `QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): static pulse equilibrium IC fill reviewed.
- `QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): advecting pulse IC fill with beta-order-dependent radiation moments reviewed.
- `problem_main()` (`reviewed`/`partial`): dual-run (static + advecting) setup, phase-aligned comparison, L1 error metric, plotting/CSV diagnostics reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroPulseDyn/testRadhydroPulseDyn.cpp`
- `PulseProblem` / `AdvPulseProblem` (`reviewed`): empty tag types for non-advecting/advecting pulse variants.
- Constants/config (`beta_order_`, `T_lo`, `T_hi`, `rho0`, `a_rad`, `c`, `width`, `erad_floor`, `mu`, `k_B`, `kappa0`, `v0_adv`) (`reviewed`): dynamic-diffusion pulse test parameters reviewed.
- `quokka::EOS_Traits<PulseProblem>` / `quokka::EOS_Traits<AdvPulseProblem>` / `RadSystem_Traits<PulseProblem>` / `RadSystem_Traits<AdvPulseProblem>` / `Physics_Traits<PulseProblem>` / `Physics_Traits<AdvPulseProblem>` specializations (`reviewed`): hydro+radiation pulse configurations reviewed.
- `compute_initial_Tgas(...)` / `compute_exact_rho(...)` (`reviewed`): Gaussian pulse temperature and equilibrium density profiles reviewed.
- `RadSystem<PulseProblem>::ComputePlanckOpacity(...)` / `RadSystem<AdvPulseProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` overloads (`reviewed`): constant-opacity closures reviewed.
- `QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): static pulse equilibrium IC fill reviewed.
- `QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): advecting pulse IC fill with beta-order-dependent radiation moments reviewed.
- `problem_main()` (`reviewed`/`partial`): dual-run (static + advecting) setup, phase-aligned comparison, L1 error metric, plotting/CSV diagnostics reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp`
- `PulseProblem` / `AdvPulseProblem` (`reviewed`): empty tag types for non-advecting/advecting pulse variants.
- Constants/config (`T_lo`, `T_hi`, `rho0`, `a_rad`, `width`, `erad_floor`, `mu`, `k_B`, `kappa0`, `v0_adv`) (`reviewed`): grey-opacity pulse test parameters reviewed.
- `quokka::EOS_Traits<PulseProblem>` / `quokka::EOS_Traits<AdvPulseProblem>` / `RadSystem_Traits<PulseProblem>` / `RadSystem_Traits<AdvPulseProblem>` / `Physics_Traits<PulseProblem>` / `Physics_Traits<AdvPulseProblem>` specializations (`reviewed`): hydro+radiation grey pulse configurations reviewed.
- `compute_initial_Tgas(...)` / `compute_exact_rho(...)` (`reviewed`): Gaussian pulse temperature and equilibrium density profiles reviewed.
- `RadSystem<PulseProblem>::ComputePlanckOpacity(...)` / `RadSystem<AdvPulseProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` overloads (`reviewed`): temperature-dependent grey opacity closures reviewed.
- `QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): static pulse equilibrium IC fill reviewed.
- `QuokkaSimulation<AdvPulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): advecting pulse IC fill reviewed.
- `problem_main()` (`finding`, diagnostics/output correctness): comment says the output should include `tNew_[0]` in the filename, but `matplotlibcpp::save(std::format("./radhydro_pulse_grey_temperature.pdf", sim2.tNew_[0]))` has no `{}` placeholder (`src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp:372-373`). The time argument is ignored and the filename is constant.
- `problem_main()` (`reviewed`): dual-run comparison, symmetry check, L1 metrics, and plotting reviewed.

### `src/problems/RadhydroPulseMGconst/testRadhydroPulseMGconst.cpp`
- `SGProblem` / `MGproblem` (`reviewed`): empty tag types for grey and multigroup pulse variants.
- Constants/config (`n_groups_`, `rad_boundaries_`, `kappa0`, `T_lo`, `T_hi`, `rho0`, `a_rad`, `width`, `Erad0`, `erad_floor`, `mu`, `h_planck`, `k_B`, `v0`, `max_time`, `max_timesteps`) (`reviewed`): multigroup-constant-opacity pulse test parameters reviewed.
- `compute_initial_Tgas(...)` / `compute_exact_rho(...)` (`reviewed`): Gaussian pulse temperature and equilibrium density profiles reviewed.
- `quokka::EOS_Traits<SGProblem>` / `Physics_Traits<SGProblem>` / `RadSystem_Traits<SGProblem>` specializations (`reviewed`): single-group comparison configuration reviewed.
- `RadSystem<SGProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): constant-opacity SG closures reviewed.
- `QuokkaSimulation<SGProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): single-group equilibrium IC fill reviewed.
- `quokka::EOS_Traits<MGproblem>` / `Physics_Traits<MGproblem>` / `RadSystem_Traits<MGproblem>` specializations (`reviewed`): multigroup constant-opacity configuration reviewed.
- `RadSystem<MGproblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): constant-opacity PPL lower-value/exponent callback reviewed.
- `QuokkaSimulation<MGproblem>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup advecting pulse IC fill reviewed.
- `problem_main()` (`reviewed`/`partial`): SG vs MG runs, phase-aligned comparison, L1 metric, and plotting reviewed; no confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp`
- `MGProblem` / `ExactProblem` (`reviewed`): empty tag types for multigroup-integrated and grey-reference variants.
- Constants/config (`n_groups_`, `opacity_model_`, `rad_boundaries_`, `export_csv`, `T_lo`, `T_hi`, `rho0`, `a_rad`, `width`, `erad_floor`, `mu`, `h_planck`, `k_B`, `kappa0`, `scaleup`, `v0_adv`, `max_time`, `max_timesteps`, `T_ref`, `nu_ref`, `coeff_`) (`reviewed`): multigroup-integrated pulse test parameters reviewed.
- `quokka::EOS_Traits<MGProblem>` / `quokka::EOS_Traits<ExactProblem>` / `Physics_Traits<MGProblem>` / `Physics_Traits<ExactProblem>` / `RadSystem_Traits<MGProblem>` / `RadSystem_Traits<ExactProblem>` specializations (`reviewed`): MG and grey-reference hydro+radiation configurations reviewed.
- `compute_initial_Tgas(...)` / `compute_exact_rho(...)` / `compute_kappa(...)` (`reviewed`): Gaussian pulse profiles and frequency/temperature-dependent opacity kernel reviewed.
- `RadSystem<MGProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): group-integrated/PPL opacity interpolation callback reviewed.
- `RadSystem<ExactProblem>::ComputePlanckOpacity(...)` / `ComputeFluxMeanOpacity(...)` (`reviewed`): grey-reference opacity closures reviewed.
- `QuokkaSimulation<MGProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): multigroup advecting pulse IC fill with diffusion-limit multigroup flux initialization reviewed.
- `QuokkaSimulation<ExactProblem>::setInitialConditionsOnGrid(...)` (`finding`, correctness): this `ExactProblem` specialization writes state using `RadSystem<MGProblem>::...` indices throughout (`src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp:286-295`) instead of `RadSystem<ExactProblem>::...`. Because `MGProblem` and `ExactProblem` have different radiation-group layouts, this can write wrong components / go out of bounds.
- `problem_main()` (`reviewed`/`partial`): MG-vs-grey runs, phase alignment, error/symmetry metrics, plotting, and CSV export reviewed; no additional confirmed bug in inspected implementation this pass.

### `src/problems/RadhydroBB/testRadhydroBB.cpp`
- `PulseProblem` (`reviewed`): empty tag type for trait specialization.
- Constants/config (`export_csv`, `n_groups_`, `rad_boundaries_`, `c`, `beta_order_`, `v0`, `kappa0`, `chat`, `T0`, `rho0`, `a_rad`, `mu`, `k_B`, `nu_unit`, `T_equilibrium`, `max_time`, `erad_floor`) (`reviewed`): blackbody-spectrum advection test parameters reviewed.
- `quokka::EOS_Traits<PulseProblem>` / `Physics_Traits<PulseProblem>` / `RadSystem_Traits<PulseProblem>` specializations (`reviewed`): multigroup hydro+radiation blackbody-advection configuration reviewed.
- `RadSystem<PulseProblem>::DefineOpacityExponentsAndLowerValues(...)` (`reviewed`): constant-opacity multigroup callback reviewed.
- `compute_exact_bb(...)` (`reviewed`): dimensionless Planck-spectrum exact solution helper reviewed.
- `QuokkaSimulation<PulseProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform advecting medium + low-radiation-floor IC fill reviewed.
- `problem_main()` (`finding`, diagnostics hygiene): contains leftover debug code (`// insert a dummy breakpoint` and `std::cout << aa`) (`src/problems/RadhydroBB/testRadhydroBB.cpp:316`, `:319`) that emits unrelated stdout during the test run.
- `problem_main()` (`reviewed`): evolve path, spectrum extraction, exact CSV load/integration, temperature/flux error metrics, plotting, and CSV exports reviewed.

### `src/problems/RayleighTaylor2D/testRayleighTaylor2D.cpp`
- `RTProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<RTProblem>` / `HydroSystem_Traits<RTProblem>` / `Physics_Traits<RTProblem>` specializations (`reviewed`): 2D hydro + passive-scalar RT configuration reviewed.
- Constants (`g_x`, `g_y`, `g_z`) (`reviewed`): body-force setup reviewed.
- `QuokkaSimulation<RTProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): stratified density/scalar IC with random perturbation and hydrostate initialization reviewed.
- `QuokkaSimulation<RTProblem>::addStrangSplitSources(...)` (`reviewed`): gravity momentum update + kinetic-energy correction source term reviewed.
- `QuokkaSimulation<RTProblem>::refineGrid(...)` (`reviewed`): density-gradient AMR tagging reviewed.
- `problem_main()` (`reviewed`): BC setup, initialize/evolve path reviewed (driver has no explicit pass/fail assertion).

### `src/problems/RayleighTaylor3D/testRayleighTaylor3D.cpp`
- `RTProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<RTProblem>` / `HydroSystem_Traits<RTProblem>` / `Physics_Traits<RTProblem>` specializations (`reviewed`): 3D hydro + passive-scalar RT configuration reviewed.
- Constants (`g_x`, `g_y`, `g_z`) (`reviewed`): body-force setup reviewed.
- `QuokkaSimulation<RTProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): stratified density/scalar IC with random vertical perturbation and hydrostate initialization reviewed.
- `QuokkaSimulation<RTProblem>::addStrangSplitSources(...)` (`reviewed`): gravity momentum update + kinetic-energy correction source term reviewed.
- `QuokkaSimulation<RTProblem>::refineGrid(...)` (`reviewed`): 3D density-gradient AMR tagging reviewed.
- `QuokkaSimulation<RTProblem>::computeAfterTimestep()` (`reviewed`): periodic 1D mixing-profile extraction and file output (`profile.txt`) reviewed.
- `problem_main()` (`reviewed`): BC setup, initialize/evolve path reviewed (driver has no explicit pass/fail assertion).

### `src/problems/SphericalCollapse/testSphericalCollapse.cpp`
- `GlobalConfig` (`reviewed`): static runtime configuration holder for particle count/seed reviewed.
- `GlobalConfig::num_particles` / `GlobalConfig::seed` definitions (`reviewed`): default initialization reviewed.
- `CollapseProblem` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<CollapseProblem>` / `Particle_Traits<CollapseProblem>` / `HydroSystem_Traits<CollapseProblem>` / `Physics_Traits<CollapseProblem>` specializations (`reviewed`): self-gravity + CIC-particle collapse configuration reviewed.
- `QuokkaSimulation<CollapseProblem>::setInitialConditionsOnGrid(...)` (`finding`, portability): unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/SphericalCollapse/testSphericalCollapse.cpp:70`, `:75`), so the specialization is not dimension-safe for 1D/2D builds.
- `QuokkaSimulation<CollapseProblem>::createInitialCICParticles()` (`finding`, robustness): computes `particle_mass = total_particle_mass / num_particles` (`src/problems/SphericalCollapse/testSphericalCollapse.cpp:104`) with no guard for `num_particles <= 0`.
- `QuokkaSimulation<CollapseProblem>::refineGrid(...)` (`reviewed`): density-threshold AMR tagging reviewed.
- `QuokkaSimulation<CollapseProblem>::ComputeDerivedVar(...)` (`reviewed`): `gpot` derived variable fill from `phi` reviewed.
- `problem_main()` (`reviewed`): runtime parameter parsing, initialize/evolve path reviewed.

### `src/problems/SN/testSN.cpp`
- `SNProblem` (`reviewed`): empty tag type for trait specialization.
- Globals/config (`refine_half_domain`, `max_Eint_global`, `max_Eint_history`, `t_history`, `SN_particles_file`, `coolingTableType_`, `mu`, `gamma_`, `CV`, `cloudy_H_mass_fraction`, `year`, `B0`, `n_amb`) (`reviewed`): SN feedback test configuration/state reviewed.
- `Particle_Traits<SNProblem>` / `quokka::EOS_Traits<SNProblem>` / `HydroSystem_Traits<SNProblem>` / `Physics_Traits<SNProblem>` / `SimulationData<SNProblem>` specializations (`reviewed`): MHD + test-particle + passive-scalar SN setup reviewed.
- `QuokkaSimulation<SNProblem>::createInitialTestParticles()` (`reviewed`): ASCII particle import and SNProgenitor/boost adjustments reviewed.
- `QuokkaSimulation<SNProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): `AMREX_D_TERM(dx[0], *dx[1], *dx[2])` usage at `src/problems/SN/testSN.cpp:152` is valid AMReX macro style (the `*` tokens are multiplication operators in macro arguments), not a dereference bug.
- `QuokkaSimulation<SNProblem>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): ambient-state interpolation, passive-scalar initialization, and boosted hydro IC fill reviewed.
- `QuokkaSimulation<SNProblem>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): uniform background magnetic field face-var initialization reviewed.
- `QuokkaSimulation<SNProblem>::refineGrid(...)` (`finding`, AMR region selection): normalized coordinates omit subtraction of `ProbLo()` (`src/problems/SN/testSN.cpp:198-200`), so the selected subregion shifts if domain lower bounds are nonzero.
- `QuokkaSimulation<SNProblem>::computeAfterTimestep()` (`reviewed`): max internal-energy tracking history reviewed.
- `problem_main()` (`reviewed`): `AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2])` volume calculations (`src/problems/SN/testSN.cpp:241`, `:253`) use valid AMReX macro syntax (multiplication-token style), not an invalid dereference.
- `problem_main()` (`finding`, test validity): Galilean-invariance error norms and pass/fail `status` update are compiled only under `#ifdef HAVE_PYTHON` (`src/problems/SN/testSN.cpp:344-404`), so non-Python builds skip the main invariance validation but still return success except for scalar checks.
- `problem_main()` (`reviewed`): runtime parameter parsing, scalar conservation/enhancement validation, baseline/boosted runs, remap alignment, and plotting reviewed.

### `src/problems/Turbulence/testTurbulence.cpp`
- `TurbulentBox` (`reviewed`): empty tag type for trait specialization.
- `Physics_Traits<TurbulentBox>` / `quokka::EOS_Traits<TurbulentBox>` / `HydroSystem_Traits<TurbulentBox>` specializations (`reviewed`): isothermal hydro turbulence-box configuration reviewed.
- `SimulationData<TurbulentBox>` specialization (`reviewed`): time/velocity-dispersion history vectors reviewed.
- `QuokkaSimulation<TurbulentBox>::setInitialConditionsOnGrid(...)` (`reviewed`): uniform-density zero-velocity IC fill with passive scalar initialization reviewed.
- `QuokkaSimulation<TurbulentBox>::refineGrid(...)` (`finding`, portability): AMR tagger unconditionally samples y/z neighbors (`state(..., j±1, ...)`, `state(..., k±1, ...)`) (`src/problems/Turbulence/testTurbulence.cpp:85`, `:87`) and computes a 3D gradient norm, so the implementation is hard-coded for 3D without a dimension guard.
- `QuokkaSimulation<TurbulentBox>::computeAfterTimestep()` (`reviewed`): velocity-dispersion diagnostic accumulation reviewed.
- `problem_main()` (`finding`, robustness): the dispersion check reads `sim.turbParams_["target_vdisp"]` via `std::stod(...)` and divides by `target_vdisp` without validating presence/nonzero value (`src/problems/Turbulence/testTurbulence.cpp:133-134`), so malformed or zero input can throw or produce invalid relative error.
- `problem_main()` (`reviewed`): periodic BC setup, evolve path, tolerance check, and optional plotting reviewed.

### `src/problems/StarCluster/testStarCluster.cpp`
- `StarCluster` (`reviewed`): empty tag type for trait specialization.
- `quokka::EOS_Traits<StarCluster>` / `HydroSystem_Traits<StarCluster>` / `Physics_Traits<StarCluster>` specializations (`reviewed`): isothermal self-gravitating cloud-collapse configuration reviewed.
- `SimulationData<StarCluster>` specialization (`reviewed`): turbulence tables and cloud parameter storage reviewed.
- `QuokkaSimulation<StarCluster>::preCalculateInitialConditions()` (`finding`, robustness): virial normalization computes `M_sphere ~ R_sphere^3`, `rms_dv_target ~ ... / R_sphere`, and `rescale_factor = rms_dv_target / rms_dv_actual` without guarding `R_sphere > 0` or `rms_dv_actual > 0` (`src/problems/StarCluster/testStarCluster.cpp:104-108`).
- `QuokkaSimulation<StarCluster>::setInitialConditionsOnGrid(...)` (`finding`, portability): the IC kernel unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/StarCluster/testStarCluster.cpp:136`, `:152`) and `dvz`, so the specialization is not dimension-safe for 1D/2D builds.
- `QuokkaSimulation<StarCluster>::refineGrid(...)` (`reviewed`): Jeans-length AMR tagging reviewed.
- `QuokkaSimulation<StarCluster>::ComputeDerivedVar(...)` (`reviewed`): `log_density` derived variable fill reviewed.
- `problem_main()` (`reviewed`): parameter parsing, initialization, and evolve path reviewed (driver has no explicit pass/fail validation).

### `src/problems/PopIII/testPopIII.cpp`
- `PopIII` (`reviewed`): empty tag type for trait specialization.
- `HydroSystem_Traits<PopIII>` / `Physics_Traits<PopIII>` specializations (`reviewed`): self-gravitating primordial-chemistry hydro configuration reviewed.
- `SimulationData<PopIII>` specialization (`reviewed`): turbulence tables, cloud parameters, and species initialization storage reviewed.
- `QuokkaSimulation<PopIII>::preCalculateInitialConditions()` (`finding`, robustness): `rms_dv_target` is initialized to `NAN`, queried from `perturb.rms_velocity`, and used in `rescale_factor = rms_dv_target / rms_dv_actual` without validation (`src/problems/PopIII/testPopIII.cpp:161-165`); missing input silently seeds NaNs into the IC velocity field.
- `QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(...)` (`finding`, portability): the IC setup unconditionally uses z-dimension geometry (`prob_lo[2]`, `prob_hi[2]`, `dx[2]`) (`src/problems/PopIII/testPopIII.cpp:245`, `:261`) without a dimension guard.
- `QuokkaSimulation<PopIII>::setInitialConditionsOnGrid(...)` (`finding`, robustness): species normalization divides by `rhotot` and then by `msum` (`src/problems/PopIII/testPopIII.cpp:277`, `:282`) with no guard; zero species abundances or `numdens_init == 0` will produce NaNs.
- `QuokkaSimulation<PopIII>::refineGrid(...)` (`reviewed`): Jeans-length + density-threshold AMR tagging reviewed.
- `QuokkaSimulation<PopIII>::ComputeDerivedVar(...)` (`reviewed`): `temperature`, `pressure`, `velx`, and `sound_speed` branches reviewed.
- `problem_main()` (`reviewed`): parameter parsing, floor setup, initialization, and evolve path reviewed.

### `src/problems/TallBoxSf/testTallBoxSf.cpp`
- `mu` constant / `TheProblem` (`reviewed`): problem tag and mean-particle-mass constant reviewed.
- `SimulationData<TheProblem>` / `Particle_Traits<TheProblem>` / `HydroSystem_Traits<TheProblem>` / `quokka::EOS_Traits<TheProblem>` / `Physics_Traits<TheProblem>` specializations (`reviewed`): tall-box SF + stochastic stellar population configuration reviewed.
- `QuokkaSimulation<TheProblem>::createInitialStochasticStellarPopParticles()` (`reviewed`): ASCII particle load and integer-component initialization path reviewed (including sparse-level particle-container iteration workaround).
- `QuokkaSimulation<TheProblem>::refineGrid(...)` (`finding`, portability): geometrical tagger unconditionally uses z-coordinate geometry (`prob_lo[2]`, `dx[2]`) (`src/problems/TallBoxSf/testTallBoxSf.cpp:145`) without a 3D guard.
- `QuokkaSimulation<TheProblem>::preCalculateInitialConditions()` (`reviewed`): turbulence-table load, IC-table CSV load, and GPU table copy path reviewed.
- `QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(...)` (`reviewed`): `AMREX_D_TERM(dx[0], *dx[1], *dx[2])` at `src/problems/TallBoxSf/testTallBoxSf.cpp:250` is valid AMReX macro multiplication-token syntax (not a dereference bug).
- `QuokkaSimulation<TheProblem>::setInitialConditionsOnGrid(...)` (`finding`, portability): IC fill unconditionally uses z-dimension geometry (`src/problems/TallBoxSf/testTallBoxSf.cpp:255`) without compile-time dimension guards.
- `QuokkaSimulation<TheProblem>::ComputeDerivedVar(...)` (`reviewed`): all derived-variable branches (`gpot`, `temperature`, `c_s`, scalar/hot/warm outflow diagnostics) reviewed.
- `QuokkaSimulation<TheProblem>::addStrangSplitSources(...)` (`finding`, portability): source-term kernel unconditionally writes/uses z-components (`posvec[2]`, `GradPhi[2]`) (`src/problems/TallBoxSf/testTallBoxSf.cpp:405`, `:422`) without a dimension guard.
- `AMRSimulation<TheProblem>::setCustomBoundaryConditions(...)` (`finding`, portability): diode BC helper unconditionally calls `setDiodeBCLo<2>` / `setDiodeBCHi<2>` (`src/problems/TallBoxSf/testTallBoxSf.cpp:455-456`), so the specialization is not dimension-safe outside 3D.
- `problem_main()` (`reviewed`): `AMREX_D_TERM(dx0[0], *dx0[1], *dx0[2])` volume calculation (`src/problems/TallBoxSf/testTallBoxSf.cpp:486`) uses valid AMReX macro multiplication-token syntax.
- `problem_main()` (`reviewed`/`partial`): RNG seed setup, runtime parameter parsing, explicit pre-calc on restart path, initialize/evolve flow, and total-energy diagnostics reviewed.

### `src/problems/DiskGalaxy/testDiskGalaxy.cpp`
- Anonymous-namespace constants (`keV_in_ergs`, `seconds_per_year`) / `DiskGalaxy` / `static_assert(AMREX_SPACEDIM == 3)` (`reviewed`): problem constants and explicit 3D compile-time guard reviewed.
- `quokka::EOS_Traits<DiskGalaxy>` / `HydroSystem_Traits<DiskGalaxy>` / `Physics_Traits<DiskGalaxy>` / `Particle_Traits<DiskGalaxy>` specializations (`reviewed`): MHD + self-gravity + particle feedback disk-galaxy configuration reviewed.
- `SimulationData<DiskGalaxy>` specialization (`reviewed`): halo profile table storage (`PinnedVector`s), parser state, and cached boundary values reviewed.
- `QuokkaSimulation<DiskGalaxy>::preCalculateInitialConditions()` (`reviewed`): halo-profile CSV ingest, cached profile endpoints, and optional `halo_vphi_expr` parser compilation reviewed.
- `QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGrid(...)` (`finding`, GPU portability/safety): host `PinnedVector` pointers are extracted via `dataPtr()` (`src/problems/DiskGalaxy/testDiskGalaxy.cpp:205-209`) and captured into an `AMREX_GPU_DEVICE` kernel (`:248`) where they are dereferenced by interpolation lambdas (`:272-325`). This relies on host-pinned memory being device-accessible and violates the repo’s GPU-lambda safety guidance.
- `QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGrid(...)` (`reviewed`/`partial`): disk+halo profile quadrature, MHD energy accounting, scalar initialization, and optional halo azimuthal parser path reviewed.
- `QuokkaSimulation<DiskGalaxy>::setInitialConditionsOnGridFaceVars(...)` (`reviewed`): toroidal magnetic-field face-variable initialization reviewed.
- `QuokkaSimulation<DiskGalaxy>::createInitialCICParticles()` (`reviewed`): ASCII particle import for CIC particles reviewed.
- `QuokkaSimulation<DiskGalaxy>::refineGrid(...)` (`reviewed`): geometric cylindrical refinement tagging with corner checks reviewed.
- `QuokkaSimulation<DiskGalaxy>::ComputeDerivedVar(...)` (`reviewed`): all derived-variable branches (`gpot`, `temperature`, `pressure`, `entropy`, `radius_sph`, `bfield_strength`, `radial_velocity`, `circular_velocity`) reviewed.
- `QuokkaSimulation<DiskGalaxy>::ComputeStatistics()` (`reviewed`): SFR, refined-region gas mass, cold-mass integral, stellar-mass, and cumulative-SN diagnostics reviewed.
- `problem_main()` (`reviewed`): reflecting hydro/face BC setup, initialization, and evolve path reviewed.

### `src/SimulationData.hpp`
- `SimulationData<problem_t>` (`reviewed`): default extension-point struct is intentionally empty; interface-only.

### `src/grid.hpp`
- `quokka::centering` / `quokka::direction` enums (`reviewed`): simple tags for grid centering and face direction.
- `quokka::face_dir_str` (`reviewed`): static name table for face directions.
- `quokka::grid` (`reviewed`): POD-style grid wrapper for `Array4`, index range, geometry arrays, and centering/direction metadata reviewed.
- `quokka::grid::grid(...)` (`reviewed`): memberwise constructor is straightforward.

### `src/physics_numVars.hpp`
- `Physics_NumVars` (`reviewed`): compile-time constant counts for hydro/radiation/dust/MHD variables; no function logic.

### `src/physics_info.hpp`
- `UnitSystem` (`reviewed`): unit-system enum (`CGS`, `CONSTANTS`, `CUSTOM`) reviewed.
- `Physics_Traits<problem_t>` (`reviewed`): default trait values for enabled physics and unit constants reviewed; specialization point.
- `Physics_Indices<problem_t>` (`reviewed`): compile-time component indexing/layout helpers for cc/fc state arrays reviewed.

### `src/hydro/HydroState.hpp`
- `quokka::HydroState<Nall, Nmass>` (`reviewed`): primitive-state container layout reviewed.
- `ConsHydro1D<N_passiveScalars>` (`reviewed`): 1D conserved-state helper container for MHD Riemann solvers reviewed.
- `SQUARE(T)` (`reviewed`): trivial helper returns `x*x`.
- `FastMagnetoSonicSpeed(...)` (`reviewed`): fast magnetosonic speed formula implementation reviewed; no new confirmed bug in this pass.

### `src/hydro/LLF.hpp`
- `quokka::Riemann::LLF(...)` (`reviewed`): hydrodynamic LLF/Rusanov solver path and passive-scalar flux packing reviewed; no new confirmed bug in this pass.

### `src/hydro/LLF_mhd.hpp`
- `quokka::Riemann::LLF_MHD(...)` (`reviewed`): MHD LLF/Rusanov solver path, state/flux assembly, and returned signal-speed pair reviewed; no new confirmed bug in this pass.

### `src/hydro/EOS.hpp`
- `quokka::EOS_Traits<problem_t>` (`reviewed`): default gamma-law EOS trait values and constants reviewed.
- `quokka::EOS<problem_t>` (`reviewed`): class interface/constants for EOS wrappers reviewed.
- `EOS<problem_t>::ComputeTgasFromEint(...)` (`reviewed`): chemistry/non-chemistry branches and unit rescaling reviewed.
- `EOS<problem_t>::ComputeEintFromTgas(...)` (`reviewed`): inverse temperature-to-energy path reviewed.
- `EOS<problem_t>::ComputeEintFromPres(...)` (`reviewed`): pressure-to-energy conversion path reviewed.
- `EOS<problem_t>::ComputeEintTempDerivative(...)` (`reviewed`): EOS temperature derivative path reviewed.
- `EOS<problem_t>::ComputeOtherDerivatives(...)` (`reviewed`): derivative tuple assembly (including chemistry branch) reviewed.
- `EOS<problem_t>::ComputePressure(...)` (`reviewed`): isothermal/chemistry/general pressure branches reviewed; zero-density guard in non-chemistry branch noted as intentional.
- `EOS<problem_t>::ComputeSoundSpeed(...)` (`reviewed`): sound-speed evaluation for isothermal and general EOS branches reviewed.
- `EOS<problem_t>::ComputeIsothermalSoundSpeed(...)` (`reviewed`): isothermal helper and gamma-dependent fallback reviewed.

### `src/hydro/HLLC.hpp`
- `quokka::Riemann::HLLC(...)` (`reviewed`): HLLC flux solver (Roe averages, nonlinear-wave correction, carbuncle correction, star-state flux selection) reviewed; no new confirmed bug in this pass.

### `src/hydro/HLLD.hpp`
- `quokka::Riemann::DELTA` (`reviewed`): degeneracy tolerance constant for HLLD solver.
- `quokka::Riemann::HLLD(...)` (`reviewed`): full HLLD solver path (wave speeds, star/double-star states, MM21 low-Mach correction, flux selection) reviewed; no new confirmed bug in this pass.

### `src/hydro/NSCBC_inflow.hpp`
- `NSCBC::detail::dQ_dx_inflow_x1_lower(...)` (`reviewed`): subsonic inflow characteristic derivative construction and scalar relaxation terms reviewed.
- `NSCBC::setInflowX1Lower(...)` (`reviewed`): high-order x1-lower inflow ghost fill (data derivative + characteristic derivative + extrapolated ghosts) reviewed.
- `NSCBC::setInflowX1LowerLowOrder(...)` (`reviewed`): low-order x1-lower inflow ghost fill with prescribed inflow state reviewed.

### `src/hydro/NSCBC_outflow.hpp`
- `NSCBC::detail::dQ_dx_outflow(...)` (`reviewed`): subsonic outflow characteristic derivative builder reviewed.
- `NSCBC::detail::transverse_xdir_dQ_data(...)` (`finding`, correctness): in the `AMREX_SPACEDIM == 3` z-derivative branch, the computed derivative is assigned to `dQ_dy_data` instead of `dQ_dz_data` (`src/hydro/NSCBC_outflow.hpp:132-137`, assignment at `:136`). This drops the z-transverse contribution and corrupts 3D x-boundary NSCBC transverse terms.
- `NSCBC::detail::transverse_ydir_dQ_data(...)` (`reviewed`): y-boundary transverse derivative helper reviewed.
- `NSCBC::detail::transverse_zdir_dQ_data(...)` (`reviewed`): z-boundary transverse derivative helper reviewed.
- `NSCBC::detail::permute_vel<...>(...)` (`reviewed`): primitive-variable velocity permutation helper for directional reuse reviewed.
- `NSCBC::detail::unpermute_vel<...>(...)` (`reviewed`): inverse velocity permutation helper reviewed.
- `NSCBC::setOutflowBoundary<...>(...)` (`reviewed`): higher-order NSCBC outflow ghost fill path reviewed.
- `NSCBC::setOutflowBoundaryLowOrder<...>(...)` (`finding`, correctness): in the `DIR == FluxDir::X3` reflecting fallback branch, `Q_im3` is assigned three times and `Q_im4`/`Q_im5` are never populated (`src/hydro/NSCBC_outflow.hpp:427-432`; repeated assignments at `:430-432`). This corrupts reflected ghost-state construction for z-boundaries in low-order mode.

### `src/hydro/hydro_system.cpp`
- No function definitions (`reviewed`). File only includes `hydro/hydro_system.hpp`.

### `src/hydro/mhd_system.cpp`
- No function definitions (`reviewed`). File only includes `mhd_system.hpp`.

### `src/hyperbolic_system.cpp`
- No solver logic (`reviewed`): include-only TU plus explicit `amrex::Array4<double>` / `amrex::Array4<const double>` template instantiations reviewed.

### `src/hyperbolic_system.hpp`
- `quokka::redoFlag` / `SlopeLimiter` enums (`reviewed`): control enums for retry signaling and reconstruction limiter choice reviewed.
- `HyperbolicSystem<problem_t>` (`reviewed`): reconstruction/update helper class template interface and implementation reviewed.
- `HyperbolicSystem::SlopeFunc<...>()`, `MC(...)`, `minmod(...)`, `minmod3(...)`, `Sweby(...)`, `median(...)` (`reviewed`): slope-limiter helper functions reviewed.
- `HyperbolicSystem::ReconstructStatesConstant<DIR>(...)` overload family (`reviewed`): `MultiFab`, `Array4`, and `Array4View` constant reconstruction paths reviewed.
- `HyperbolicSystem::ReconstructStatesPLM<DIR>(...)` overload family (`reviewed`): limiter-dispatched PLM reconstruction wrappers and per-cell kernel overloads reviewed.
- `HyperbolicSystem::ReconstructStatesPPM<DIR>(...)` overload family (`reviewed`): PPM reconstruction wrappers/per-cell kernels reviewed.
- `HyperbolicSystem::MonotonizeEdges(...)`, `ComputeSteepPPM<DIR>(...)`, `ComputeWENOMoments<DIR>(...)`, `ComputeWENO<DIR>(...)` (`reviewed`): PPM/WENO support routines reviewed.
- `HyperbolicSystem::ReconstructStatesPPM_EP<DIR>(...)` overload family (`reviewed`): extremum-preserving PPM reconstruction wrappers/per-cell kernels reviewed.
- `HyperbolicSystem::PredictStep(...)` (`reviewed`): predictor update with flux divergence accumulation and state-validity/redo handling reviewed; no new confirmed bug in this pass.
- `HyperbolicSystem::AddFluxesRK2(...)` (`reviewed`): RK2 corrector accumulation and redo handling reviewed; no new confirmed bug in this pass.

### `src/radiation/planck_integral.hpp`
- Constants (`USE_SECOND_ORDER`, `PI`, `gInf`, `INTERP_SIZE`, `LOG_X_MIN`, `LOG_X_MAX`, `Y_INTERP_MIN`) (`reviewed`): interpolation table configuration/constants reviewed.
- `interpolate_planck_integral(Real)` (`reviewed`): table lookup and linear/quadratic interpolation path reviewed (table values intentionally not exhaustively re-verified).
- `integrate_planck_from_0_to_x(Real)` (`reviewed`): low-`x` asymptotic approximation + interpolation dispatch + clamp/assert logic reviewed.

### `src/radiation/radiation_dust_system.hpp`
- `RadSystem<problem_t>::DefinePhotoelectricHeatingE1Derivative(...)` (`reviewed`): default PE derivative stub returns `0`.
- `RadSystem<problem_t>::ComputeJacobianForGasAndDust(...)` (`reviewed`): coupled gas-dust-radiation Jacobian assembly reviewed.
- `RadSystem<problem_t>::ComputeJacobianForGasAndDustDecoupled(...)` (`reviewed`): decoupled Jacobian assembly reviewed.
- `RadSystem<problem_t>::ComputeJacobianForGasAndDustWithPE(...)` (`reviewed`): PE-heating-augmented Jacobian assembly reviewed.
- `RadSystem<problem_t>::SolveLinearEqsWithLastColumn(...)` (`reviewed`): specialized linear solve for first-row/first-column/diagonal matrix structure reviewed.
- `RadSystem<problem_t>::SolveGasDustRadiationEnergyExchange(...)` (`reviewed`): main Newton solve path (including later Newton/update/convergence and post-iteration cooling sections) reviewed; no new confirmed bug in this pass.
- `RadSystem<problem_t>::SolveGasDustRadiationEnergyExchangeWithPE(...)` (`reviewed`): PE-heating variant Newton solve path (including later Newton/update/convergence and post-iteration cooling sections) reviewed; no new confirmed bug in this pass.
