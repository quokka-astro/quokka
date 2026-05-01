# Manual Function Review Log

This log tracks manual line-by-line review of named, non-GCC-internal function locations from `issues/ast-function-index.md`.

Status values:

- `reviewed`: function body read manually; no new high-priority correctness bug found.
- `reported`: function body read manually; issue report exists.
- `declaration`: declaration-only or no executable body at that source location.

## Progress

- Total named non-internal AST function locations: 1,032
- Manually reviewed in this log: 1,032
- Remaining: 0

## Reviewed Functions

## `src/Factory.H`

- L72 `create`: reviewed
- L89 `contains`: reviewed
- L95 `print`: reviewed
- L113 `add_sub_type`: reviewed
- L134 `key_exists_or_error`: reviewed
- L145 `table`: reviewed

## `src/QuokkaSimulation.hpp`

- L242 `initialize`: declaration
- L365 `callInOrder`: reviewed
- L433 `defineComponentNames`: reviewed
- L555 `getScalarVariableNames`: reviewed
- L571 `readParmParse`: reviewed
- L731 `rereadRuntimeParameters`: reviewed
- L746 `computeNumberOfRadiationSubsteps`: reported in `issues/radiation-cfl-nonpositive-substeps.md`
- L758 `computeMaxSignalLocal`: reviewed
- L793 `printCellProperties`: reviewed
- L819 `CheckHydroStates`: reviewed
- L830 `checkHydroStates`: reviewed
- L846 `preCalculateInitialConditions`: reviewed
- L852 `setInitialConditionsOnGrid`: reviewed
- L858 `setInitialConditionsOnGridFaceVars`: reviewed
- L867 `createInitialRadParticles`: reviewed
- L874 `createInitialCICParticles`: reviewed
- L881 `createInitialCICRadParticles`: reviewed
- L888 `createInitialStochasticStellarPopParticles`: reviewed
- L897 `createInitialSinkParticles`: reviewed
- L906 `createInitialTestParticles`: reviewed
- L916 `computeBeforeTimestep`: reviewed
- L921 `computeAfterTimestep`: reviewed
- L926 `computeAfterLevelAdvance`: reviewed
- L931 `addStrangSplitSources`: reviewed
- L937 `computePhotoelectricHeatingRate`: reviewed
- L961 `computeExternalHeatingRate`: reviewed
- L973 `addStrangSplitSourcesWithBuiltin`: reviewed
- L1045 `ComputeDerivedVar`: reviewed
- L1054 `ComputeDensityFloorDebug`: reviewed
- L1109 `ComputeStatistics`: reviewed
- L1116 `refineGrid`: reviewed
- L1122 `ErrorEst`: reviewed
- L1135 `computeReferenceSolution`: reviewed
- L1142 `computeReferenceSolution_fc`: reviewed
- L1149 `print_multifab_fc`: reviewed
- L1158 `densityFloor`: reviewed
- L1165 `computeComponentErrors`: reviewed
- L1280 `computeErrorNorm`: reviewed
- L1336 `computeAfterEvolve`: reviewed
- L1387 `advanceSingleTimestepAtLevel`: reviewed
- L1461 `fillPoissonRhsAtLevel`: reviewed
- L1477 `applyPoissonGravityAtLevel`: reviewed
- L1514 `projectFaceCenteredMagneticField`: reviewed
- L1742 `updateInitialMagneticEnergyFromFaceField`: reviewed
- L1800 `postInitialization`: reviewed
- L1820 `ApplyHydroStateFixup`: reviewed
- L1846 `FixupState`: reviewed
- L1858 `FillPatch`: reviewed
- L1885 `PreInterpState`: reviewed
- L1904 `PostInterpState`: reviewed
- L1925 `computeAxisAlignedProfile`: reviewed
- L1967 `advanceHydroAtLevelWithRetries`: reviewed
- L2089 `isCflViolated`: reviewed
- L2112 `printCoordinates`: reviewed
- L2130 `advanceHydroAtLevel`: reviewed
- L2483 `replaceFluxes`: reviewed
- L2529 `replaceEMFs`: reviewed
- L2588 `addFluxArrays`: reviewed
- L2601 `expandFluxArrays`: reviewed
- L2621 `computeHydroFluxes`: reviewed
- L2742 `computeCCPerpBfieldComps`: reviewed
- L2776 `hydroFluxFunction`: reviewed
- L2831 `computeFOHydroFluxes`: reviewed
- L2888 `hydroFOFluxFunction`: reviewed
- L2917 `swapRadiationState`: reviewed
- L2924 `subcycleRadiationAtLevel`: reviewed
- L3146 `advanceRadiationForwardEuler`: reviewed
- L3200 `advanceRadiationMidpointRK2`: reviewed
- L3271 `computeRadiationFluxes`: reviewed
- L3302 `fluxFunction`: reviewed
- L3365 `WriteSingleLevelPlotfileSimplified`: reviewed

## `src/chemistry/Chemistry.cpp`

- L17 `chemburner`: reviewed

## `src/chemistry/Chemistry.hpp`

- L29 `chemburner`: declaration
- L31 `computeChemistry`: reviewed

## `src/cooling/PhotoelectricHeating.hpp`

- L32 `const_tables`: reviewed
- L38 `is_initialized`: reviewed
- L49 `PeHeatingFromSfh`: reviewed
- L87 `PeHeatingFromConstSfr`: reviewed

## `src/cooling/ResampledCooling.cpp`

- L24 `readResampledData`: reviewed
- L69 `const_tables`: reviewed

## `src/cooling/ResampledCooling.hpp`

- L64 `const_tables`: declaration
- L68 `resampled_cooling_function`: reviewed
- L81 `ComputeTgasFromEgas`: reviewed
- L93 `ComputeEgasFromTgas`: reviewed
- L118 `ComputeCoolingLength`: reviewed
- L134 `ComputePressureFromRhoEint`: reviewed
- L147 `ComputeEntropyFromRhoEint`: reviewed
- L160 `ComputeSoundSpeedFromRhoEint`: reviewed
- L176 `ResampledCoolingFunctor`: reviewed
- L189 `operator()`: reviewed
- L198 `computeCooling`: reviewed
- L272 `readResampledData`: declaration

## `src/dust/DustDrag.hpp`

- L72 `ComputeReciprocalStoppingTime`: reviewed
- L85 `ComputeReciprocalStoppingTimeKwok`: reviewed
- L110 `computeDustDrag`: reviewed

## `src/dust/dustRiemannSolver.hpp`

- L14 `dustRiemannSolver`: reviewed

## `src/dust/dust_system.hpp`

- L63 `ComputeDustFluxes`: reviewed

## `src/hydro/EOS.hpp`

- L93 `ComputeTgasFromEint`: reviewed
- L135 `ComputeEintFromTgas`: reviewed
- L178 `ComputeEintFromPres`: reviewed
- L219 `ComputeEintTempDerivative`: reviewed
- L263 `ComputeOtherDerivatives`: reviewed
- L320 `ComputePressure`: reviewed
- L371 `ComputeSoundSpeed`: reviewed
- L416 `ComputeIsothermalSoundSpeed`: reviewed

## `src/hydro/HLLC.hpp`

- L22 `HLLC`: reviewed

## `src/hydro/HLLD.hpp`

- L21 `HLLD`: reviewed

## `src/hydro/HydroState.hpp`

- L38 `SQUARE`: reviewed
- L41 `FastMagnetoSonicSpeed`: reviewed

## `src/hydro/LLF.hpp`

- L16 `LLF`: reviewed

## `src/hydro/LLF_mhd.hpp`

- L16 `LLF_MHD`: reported in `issues/llf-mhd-passive-scalar-flux-zero.md`

## `src/hydro/NSCBC_inflow.hpp`

- L25 `dQ_dx_inflow_x1_lower`: reviewed
- L107 `setInflowX1Lower`: reviewed
- L180 `setInflowX1LowerLowOrder`: reviewed

## `src/hydro/NSCBC_outflow.hpp`

- L25 `dQ_dx_outflow`: reviewed
- L110 `transverse_xdir_dQ_data`: reported in `issues/nscbc-outflow-transverse-x3-copy-paste.md`
- L144 `transverse_ydir_dQ_data`: reviewed
- L177 `transverse_zdir_dQ_data`: reviewed
- L210 `permute_vel`: reviewed
- L236 `unpermute_vel`: reviewed
- L263 `setOutflowBoundary`: reviewed
- L365 `setOutflowBoundaryLowOrder`: reported in `issues/nscbc-outflow-transverse-x3-copy-paste.md`

## `src/hydro/hydro_system.hpp`

- L158 `GetGradFixedPotential`: declaration
- L177 `ComputeFirstOrderFluxes`: declaration
- L189 `is_eos_isothermal`: reviewed
- L195 `ConservedToPrimitive`: reviewed
- L337 `maxSignalSpeedLocal`: reviewed
- L380 `ComputeMaxSignalSpeed`: reviewed
- L459 `CheckStatesValid`: reviewed
- L517 `ComputePrimVars`: reviewed
- L552 `ComputeConsVars`: reviewed
- L575 `ComputePressure`: reviewed
- L603 `ComputeInternalEnergy`: reviewed
- L619 `ComputeSoundSpeed`: reviewed
- L637 `ComputeIsothermalSoundSpeed`: reviewed
- L659 `ComputeMagneticEnergy`: reviewed
- L681 `ComputePlasmaBeta`: reviewed
- L698 `ComputeVelocityX1`: reviewed
- L707 `ComputeVelocityX2`: reviewed
- L716 `ComputeVelocityX3`: reviewed
- L725 `isStateValid`: reviewed
- L757 `ComputeRhsFromFluxes`: reviewed
- L784 `PredictStep`: reviewed
- L808 `AddFluxesRK2`: reviewed
- L838 `ComputeFlatteningCoefficients`: reviewed
- L938 `FlattenShocks`: reviewed
- L1008 `EnforceLimits`: reviewed
- L1114 `AddInternalEnergyPdV`: reviewed
- L1170 `SyncDualEnergy`: reviewed
- L1232 `ComputeFluxes`: reported in `issues/hydro-face-velocity-upwind-density-reversed.md`

## `src/hydro/mhd_system.hpp`

- L26 `amrex_get_enum_traits`: reviewed
- L32 `amrex_get_enum_traits`: reviewed
- L36 `MinimumHydroRiemannGhost`: reviewed
- L109 `ComputeEMF`: reviewed
- L130 `AverageEMF`: reviewed
- L147 `ComputeEMF_FelkerStone2017`: reviewed
- L371 `ComputeEMF_Quokka2026`: reviewed
- L507 `ComputeEMF_Balsara2025`: reviewed
- L708 `EMFAverage_LondrilloDelZanna2004`: reviewed
- L777 `EMFAverage_Balsara2025`: reviewed
- L876 `ReconstructTo`: reviewed
- L950 `SolveInductionEqn`: reviewed

## `src/hyperbolic_system.hpp`

- L41 `amrex_get_enum_traits`: reviewed
- L50 `SlopeFunc`: reviewed
- L65 `MC`: reviewed
- L67 `minmod`: reviewed
- L72 `minmod3`: reviewed
- L80 `Sweby`: reviewed
- L88 `median`: reviewed
- L92 `AssertReconstructionRanges`: reviewed
- L253 `ReconstructStatesConstant`: reviewed
- L273 `ReconstructStatesConstant`: reviewed
- L298 `ReconstructStatesConstant`: reviewed
- L315 `ReconstructStatesPLM`: reviewed
- L344 `ReconstructStatesPLM`: reviewed
- L373 `ReconstructStatesPLM`: reviewed
- L396 `ReconstructStatesPPM`: reviewed
- L416 `ReconstructStatesPPM`: reviewed
- L506 `MonotonizeEdges`: reviewed
- L523 `ComputeSteepPPM`: reviewed
- L539 `ComputeWENOMoments`: reviewed
- L586 `ComputeWENO`: reviewed
- L601 `ReconstructStatesPPM_EP`: reviewed
- L624 `ReconstructStatesPPM_EP`: reviewed
- L645 `ReconstructStatesPPM_EP`: reviewed
- L713 `PredictStep`: reviewed
- L754 `AddFluxesRK2`: reviewed

## `src/io/DerivedFieldBase.H`

- L23 `base_identifier`: reviewed
- L25 `init`: declaration
- L26 `prepare`: declaration
- L29 `addVars`: declaration
- L34 `computeField`: declaration
- L36 `hasField`: declaration

## `src/io/DerivedFieldBase.cpp`

- L8 `init`: reviewed
- L10 `prepare`: reviewed
- L15 `addVars`: reviewed
- L22 `computeField`: reviewed
- L28 `hasField`: reviewed

## `src/io/DerivedParticleDeposition.H`

- L17 `identifier`: reviewed

## `src/io/DerivedParticleDeposition.cpp`

- L15 `isSupportedParticleType`: reviewed
- L21 `init`: reviewed
- L93 `computeField`: reviewed
- L114 `getFieldName`: reviewed

## `src/io/DiagBase.H`

- L20 `base_identifier`: reviewed
- L22 `init`: declaration
- L24 `close`: declaration
- L26 `needUpdate`: reviewed
- L28 `doDiag`: declaration
- L30 `prepare`: declaration
- L40 `addVars`: declaration
- L42 `getFieldIndex`: declaration
- L44 `getFieldIndexVec`: declaration
- L48 `setDiagData`: reviewed
- L84 `getSim`: reviewed

## `src/io/DiagBase.cpp`

- L4 `init`: reviewed
- L35 `prepare`: reviewed
- L53 `doDiag`: reviewed
- L98 `addVars`: reviewed
- L106 `getFieldIndex`: reviewed
- L121 `getFieldIndexVec`: reviewed

## `src/io/DiagFilter.H`

- L16 `init`: declaration
- L17 `setup`: declaration

## `src/io/DiagFilter.cpp`

- L4 `init`: reviewed
- L34 `setup`: reviewed

## `src/io/DiagPDF.H`

- L11 `identifier`: reviewed
- L13 `init`: declaration
- L15 `prepare`: declaration
- L21 `addVars`: declaration
- L23 `MFVecMin`: declaration
- L24 `MFVecMax`: declaration
- L25 `writePDFToFile`: declaration
- L27 `close`: reviewed
- L45 `getIdxVec`: declaration
- L54 `getBinIndex1D`: reviewed
- L62 `getTotalBinCount`: reviewed
- L72 `processDiag`: reported in `issues/diagpdf-periodic-finemask-double-count.md`

## `src/io/DiagPDF.cpp`

- L14 `init`: reviewed
- L56 `addVars`: reviewed
- L68 `prepare`: reviewed
- L86 `getIdxVec`: reviewed
- L103 `MFVecMin`: reviewed
- L115 `MFVecMax`: reviewed
- L127 `writePDFToFile`: reviewed

## `src/io/DiagFramePlane.H`

- L20 `identifier`: reviewed
- L22 `init`: declaration
- L24 `prepare`: declaration
- L30 `addVars`: declaration
- L33 `getParticleTypes`: reviewed
- L39 `VisMF2D`: declaration
- L41 `Write2DMFHeader`: declaration
- L43 `Find2FOffsets`: declaration
- L46 `write_2D_header`: declaration
- L48 `Write2DPlotfileHeader`: declaration
- L54 `close`: reviewed
- L80 `processDiag`: reviewed

## `src/io/DiagFramePlane.cpp`

- L16 `printLowerDimIntVect`: reviewed
- L32 `printLowerDimBox`: reviewed
- L43 `init`: reviewed
- L96 `addVars`: reviewed
- L104 `prepare`: reviewed
- L213 `Write2DMultiLevelPlotfile`: reviewed
- L284 `Write2DPlotfileHeader`: reviewed
- L351 `VisMF2D`: reported in `issues/vismf2d-byteswritten-accumulates.md`
- L480 `Write2DMFHeader`: reviewed
- L543 `Find2FOffsets`: reviewed
- L617 `write_2D_header`: reviewed

## `src/io/DiagParticleTxt.H`

- L15 `identifier`: reviewed
- L17 `init`: declaration
- L19 `prepare`: declaration
- L22 `addVars`: declaration
- L24 `close`: reviewed
- L34 `processDiag`: reported in `issues/diagparticletxt-empty-particles-skips-output.md`

## `src/io/DiagParticleTxt.cpp`

- L5 `init`: reviewed
- L39 `prepare`: reviewed
- L47 `addVars`: reviewed

## `src/io/DiagPlotfile.H`

- L31 `identifier`: reviewed
- L33 `init`: declaration
- L35 `prepare`: declaration
- L38 `addVars`: declaration
- L40 `close`: reviewed
- L46 `getDiagFileName`: reviewed
- L49 `getParticleTypes`: reviewed
- L62 `processDiag`: reviewed
- L181 `WriteMetadataFile`: reviewed

## `src/io/DiagPlotfile.cpp`

- L5 `init`: reviewed
- L67 `prepare`: reviewed
- L75 `addVars`: reviewed

## `src/io/DiagProjectionPlot.H`

- L25 `identifier`: reviewed
- L27 `init`: declaration
- L29 `prepare`: declaration
- L32 `addVars`: declaration
- L34 `close`: reviewed
- L40 `getParticleTypes`: reviewed
- L49 `processDiag`: reviewed

## `src/io/DiagProjectionPlot.cpp`

- L5 `init`: reviewed
- L95 `prepare`: reviewed
- L117 `addVars`: reviewed

## `src/io/io_utils.hpp`

- L18 `ScopedVisMFNOutFiles`: reviewed
- L24 `~ScopedVisMFNOutFiles`: reviewed
- L31 `ScopedVisMFNOutFiles`: declaration
- L32 `operator=`: declaration
- L33 `ScopedVisMFNOutFiles`: declaration
- L34 `operator=`: declaration

## `src/io/openPMD.cpp`

- L32 `getReversedVec`: reviewed
- L47 `getReversedVec`: reviewed
- L57 `SetupMeshComponent`: reviewed
- L80 `GetMeshComponentName`: reviewed
- L94 `WriteFile`: reviewed

## `src/io/openPMD.hpp`

- L25 `getReversedVec`: declaration
- L26 `getReversedVec`: declaration
- L27 `SetupMeshComponent`: declaration
- L28 `GetMeshComponentName`: declaration
- L33 `WriteFile`: declaration

## `src/io/projection.cpp`

- L30 `direction_to_string`: reviewed
- L50 `printLowerDimIntVect`: reviewed
- L66 `printLowerDimBox`: reviewed
- L69 `Write2DMultiLevelPlotfile`: reviewed
- L118 `Write2DPlotfileHeader`: reviewed
- L185 `VisMF2D`: reported in `issues/vismf2d-byteswritten-accumulates.md`
- L314 `Write2DMFHeader`: reviewed
- L377 `Find2FOffsets`: reviewed
- L451 `write_2D_header`: reviewed
- L461 `transform_box_to_2D`: reviewed
- L490 `transform_realbox_to_2D`: reviewed
- L519 `transform_ref_ratio_to_2D`: reviewed
- L541 `WriteProjection`: reviewed

## `src/io/projection.hpp`

- L43 `direction_to_string`: declaration
- L44 `transform_box_to_2D`: declaration
- L45 `transform_realbox_to_2D`: declaration
- L47 `printLowerDimIntVect`: declaration
- L48 `printLowerDimBox`: declaration
- L50 `Write2DMultiLevelPlotfile`: declaration
- L54 `Write2DPlotfileHeader`: declaration
- L59 `VisMF2D`: declaration
- L61 `Write2DMFHeader`: declaration
- L63 `Find2FOffsets`: declaration
- L66 `write_2D_header`: declaration
- L70 `ComputePlaneProjectionFromMultiFab`: reviewed
- L194 `WriteProjection`: declaration
- L201 `WriteProjection`: reviewed

## `src/main.hpp`

- L17 `main`: declaration
- L18 `problem_main`: declaration

## `src/main.cpp`

- L18 `main`: reviewed

## `src/linear_advection/linear_advection.hpp`

- L50 `ComputeMaxSignalSpeed`: reviewed
- L66 `ConservedToPrimitive`: reviewed
- L77 `isStateValid`: reviewed
- L87 `PredictStep`: reviewed
- L124 `AddFluxesRK2`: reviewed
- L169 `ComputeFluxes`: reviewed

## `src/linear_advection/AdvectionSimulation.hpp`

- L66 `AdvectionSimulation`: reviewed
- L67 `AdvectionSimulation`: reviewed
- L69 `initialize`: reviewed
- L75 `setCustomGhostCells`: reviewed
- L162 `computeMaxSignalLocal`: reviewed
- L174 `printCellProperties`: reviewed
- L179 `fillPoissonRhsAtLevel`: reviewed
- L184 `applyPoissonGravityAtLevel`: reviewed
- L189 `preCalculateInitialConditions`: reviewed
- L195 `setInitialConditionsOnGrid`: reviewed
- L201 `setInitialConditionsOnGridFaceVars`: reviewed
- L210 `createInitialRadParticles`: reviewed
- L217 `createInitialCICParticles`: reviewed
- L224 `createInitialCICRadParticles`: reviewed
- L231 `createInitialStochasticStellarPopParticles`: reviewed
- L237 `createInitialSinkParticles`: reviewed
- L243 `createInitialTestParticles`: reviewed
- L250 `computeBeforeTimestep`: reviewed
- L255 `computeAfterTimestep`: reviewed
- L260 `ComputeDerivedVar`: reviewed
- L265 `ComputeStatistics`: reviewed
- L272 `refineGrid`: reviewed
- L278 `ErrorEst`: reviewed
- L284 `FixupState`: reviewed
- L290 `computeReferenceSolution`: reviewed
- L297 `computeAfterEvolve`: reviewed
- L319 `advanceSingleTimestepAtLevel`: reviewed
- L436 `computeFluxes`: reviewed
- L484 `fluxFunction`: reviewed
- L513 `WriteSingleLevelPlotfileSimplified`: reviewed

## `src/math/FastMath.hpp`

- L34 `fastlg`: reviewed
- L42 `fastpow2`: reviewed
- L50 `lg`: reviewed
- L56 `pow2`: reviewed
- L58 `log10`: reviewed
- L64 `pow10`: reviewed
- L74 `inverse_pow2`: reviewed

## `src/math/Interpolate2D.hpp`

- L15 `interpolate2d`: reported in `issues/interpolate2d-upper-y-boundary.md`

## `src/math/ODEIntegrate.hpp`

- L24 `rk12_single_step`: reviewed
- L61 `rk23_single_step`: reviewed
- L118 `error_norm`: reported in `issues/ode-error-norm-negative-state.md`
- L137 `rk_adaptive_integrate`: reviewed

## `src/math/interpolate.hpp`

- L112 `interpolate_arrays`: reviewed
- L139 `interpolate_value`: reviewed

## `src/math/math_impl.hpp`

- L15 `clamp`: reviewed
- L18 `sgn`: reviewed

## `src/math/quadrature.hpp`

- L14 `kernel_wendland_c2`: reviewed
- L26 `quad_3d`: reviewed
- L72 `quad_2d`: reviewed
- L79 `quad_1d`: reviewed

## `src/math/root_finding.hpp`

- L66 `eps_tolerance`: reviewed
- L68 `eps_tolerance`: reviewed
- L71 `eps_tolerance`: reviewed
- L78 `operator()`: reviewed
- L96 `bracket`: reviewed
- L150 `safe_div`: reviewed
- L167 `secant_interpolate`: reviewed
- L189 `quadratic_interpolate`: reviewed
- L237 `cubic_interpolate`: reviewed
- L279 `toms748_solve`: reviewed
- L441 `toms748_solve`: reviewed

## `src/math/spherical_geometry.hpp`

- L19 `minDistSqToInterval`: reviewed
- L30 `maxDistSqToInterval`: reviewed
- L39 `addPointUnique`: reviewed
- L57 `planeBoxSectionArea`: reviewed
- L212 `sphericalSectionAreaInCell`: reviewed

## `src/math/gauss.hpp`

- L28 `get_value`: reviewed
- L64 `abscissa`: reviewed
- L74 `weights`: reviewed
- L89 `abscissa`: reviewed
- L99 `weights`: reviewed
- L114 `abscissa`: reviewed
- L124 `weights`: reviewed
- L165 `abscissa`: reviewed
- L172 `weights`: reviewed
- L184 `abscissa`: reviewed
- L191 `weights`: reviewed
- L203 `abscissa`: reviewed
- L211 `weights`: reviewed
- L246 `abscissa`: reviewed
- L254 `weights`: reviewed
- L267 `abscissa`: reviewed
- L275 `weights`: reviewed
- L288 `abscissa`: reviewed
- L297 `weights`: reviewed
- L335 `abscissa`: reviewed
- L343 `weights`: reviewed
- L356 `abscissa`: reviewed
- L364 `weights`: reviewed
- L377 `abscissa`: reviewed
- L387 `weights`: reviewed
- L428 `abscissa`: reviewed
- L436 `weights`: reviewed
- L449 `abscissa`: reviewed
- L458 `weights`: reviewed
- L472 `abscissa`: reviewed
- L483 `weights`: reviewed
- L527 `abscissa`: reviewed
- L536 `weights`: reviewed
- L550 `abscissa`: reviewed
- L559 `weights`: reviewed
- L573 `abscissa`: reviewed
- L584 `weights`: reviewed
- L634 `abscissa`: reviewed
- L636 `weights`: reviewed
- L638 `integrate`: reviewed
- L666 `integrate`: reviewed

## `src/util/ArrayUtil.hpp`

- L13 `strided_vector_from`: reviewed

## `src/util/ArrayView_2d.hpp`

- L21 `reorderMultiIndex`: declaration
- L23 `reorderMultiIndex`: reviewed
- L25 `reorderMultiIndex`: reviewed

## `src/util/ArrayView_3d.hpp`

- L22 `reorderMultiIndex`: declaration
- L24 `reorderMultiIndex`: reviewed
- L26 `reorderMultiIndex`: reviewed
- L28 `reorderMultiIndex`: reviewed

## `src/util/BC.hpp`

- L45 `amrex_get_enum_traits`: reviewed
- L85 `isNormalComponent`: reviewed
- L131 `BC`: reviewed
- L163 `BC`: reviewed
- L167 `BC_cc`: reviewed
- L199 `BC_fc`: reviewed

## `src/util/CheckNaN.hpp`

- L18 `CheckSymmetryArray`: reviewed
- L25 `CheckSymmetryFluxes`: reviewed
- L33 `CheckNaN`: reviewed

## `src/util/DataTable.hpp`

- L35 `amrex_get_enum_traits`: reviewed
- L38 `amrex_get_enum_traits`: reviewed
- L96 `find_interpolation_data`: reported in `issues/datatable-single-point-dimension.md`
- L148 `interpolate`: reviewed
- L218 `interpolate_single`: reviewed
- L301 `interpolate_from_indices`: reviewed
- L320 `interpolate_single_from_indices`: reviewed
- L450 `ioProcessorNumber`: reviewed
- L452 `bcastScalar`: reviewed
- L454 `bcastArray`: reviewed
- L459 `bcastString`: reviewed
- L471 `bcastStringArray`: reviewed
- L478 `bcastVector`: reviewed
- L490 `bcastSpacingType`: reviewed
- L499 `bcastSpacingTypes`: reviewed
- L515 `flatDataSize`: reviewed
- L524 `flatDataIndex`: reviewed
- L533 `setMetadata`: reviewed
- L574 `initialize`: reviewed
- L582 `initialize`: reviewed
- L602 `initialize`: reviewed
- L631 `initialize`: reviewed
- L665 `initialize`: reviewed
- L706 `const_tables`: reviewed
- L734 `is_initialized`: reviewed
- L752 `sizes`: reviewed
- L755 `size`: reviewed
- L763 `num_outputs`: reviewed
- L766 `input_names`: reviewed
- L767 `output_names`: reviewed
- L768 `input_units`: reviewed
- L769 `output_units`: reviewed
- L772 `input_name`: reviewed
- L778 `output_name`: reviewed
- L784 `input_unit`: reviewed
- L790 `output_unit`: reviewed
- L798 `initializeStorage`: reported in `issues/datatable-single-point-dimension.md`
- L849 `fillDataTables`: reviewed
- L886 `fillDataTablesFlat`: reviewed
- L927 `initializeCommonFlat`: reviewed
- L937 `initialize_common`: reviewed
- L946 `initialize_common`: reviewed
- L998 `CSVReader`: reported in `issues/datatable-single-point-dimension.md`
- L1282 `H5Reader`: reported in `issues/datatable-single-point-dimension.md`

## `src/util/fextract.hpp`

- L12 `fextract`: declaration

## `src/util/fextract.cpp`

- L17 `fextract`: reviewed

## `src/util/Optional.hpp`

Reviewed class special member functions and operators: default/value/copy/move constructors, copy/move assignment, destructor, boolean conversion, and dereference operator. No high-priority correctness issue found.

## `src/util/richardson.hpp`

- L35 `applyQuietDefaults`: reviewed
- L52 `run`: reviewed

## `src/util/time_units.hpp`

Reviewed `registerTimeUnitConstants`; it is inline and not present in the non-internal AST manifest. No high-priority correctness issue found.

## `src/util/valarray.hpp`

- L53 `size`: reviewed
- L55 `fillin`: reviewed
- L62 `hasnan`: reviewed
- L207 `abs`: reviewed
- L217 `min`: reviewed
- L229 `max`: reviewed
- L241 `sum`: reviewed

Reviewed `valarray` constructors, element accessors, arithmetic operators, comparison operators, and compound assignment operators. No high-priority correctness issue found.

## `src/particles/particle_types.hpp`

- L11 `bitflag`: reviewed
- L31 `operator|`: reviewed
- L36 `operator&`: reviewed
- L96 `amrex_get_enum_traits`: reviewed
- L106 `amrex_get_enum_traits`: reviewed
- L134 `amrex_get_enum_traits`: reviewed
- L157 `amrex_get_enum_traits`: reviewed
- L203 `amrex_get_enum_traits`: reviewed
- L222 `amrex_get_enum_traits`: reviewed
- L269 `amrex_get_enum_traits`: reviewed
- L280 `amrex_get_enum_traits`: reviewed
- L314 `amrex_get_enum_traits`: reviewed
- L347 `expandEnumNames`: reviewed
- L381 `getParticleRealCompNames`: reviewed
- L400 `getParticleIntCompNames`: reviewed
- L427 `get_units_data`: reviewed
- L533 `particleParmParse`: reported in `issues/stochastic-star-low-mass-cap-nonpositive.md`

## `src/particles/particle_utils.hpp`

- L101 `computeJeansDensity`: reviewed
- L124 `computePlasmaBeta`: reviewed
- L132 `roundoffMultiFab`: reviewed

## `src/particles/particle_update.hpp`

- L24 `updateParticleProperties`: reviewed
- L39 `applyUpdate`: reviewed
- L65 `updateProperties`: reviewed
- L73 `updateParticleProperties`: reviewed
- L88 `updateParticleProperties`: reviewed
- L101 `updateProperties`: reviewed

## `src/particles/particle_radiation.hpp`

- L26 `const_tables`: reviewed
- L32 `is_initialized`: reviewed
- L45 `updateLuminosity`: reviewed

## `src/particles/particle_destruction.hpp`

- L16 `destroyParticlesImpl`: reviewed
- L101 `ParticleChecker`: reviewed
- L106 `operator()`: reviewed
- L116 `destroyParticles`: reviewed

## `src/particles/particle_deposition.hpp`

- L28 `NearestEight`: reviewed
- L67 `RadDeposition::operator()`: reviewed
- L107 `ParticleMassDensityDeposition::operator()`: reviewed
- L135 `depositParticleMassDensity`: reviewed
- L195 `MassDeposition::operator()`: reviewed
- L220 `DepositionCount::operator()`: reviewed
- L238 `depositThermalSNR`: reviewed
- L295 `depositThermalKineticMomentumSNR`: reviewed
- L440 `depositToBuffer`: reviewed
- L631 `addCompositeBufferToState`: reviewed
- L720 `addThermalOnlyBufferToState`: reviewed
- L767 `addBufferToState`: reviewed
- L800 `updateEvolutionStageAndDeathDensity`: reviewed
- L851 `updateEvolutionStage`: reviewed
- L887 `SNDeposition`: reviewed

## `src/particles/particle_creation.hpp`

- L24 `createParticlesImpl`: reviewed
- L142 `ParticleChecker`: reviewed
- L145 `operator()`: reviewed
- L158 `ParticleCreator`: reviewed
- L173 `operator()`: reviewed
- L182 `createParticles`: reviewed
- L205 `checkSinkCreation`: reviewed
- L268 `initializeSinkLikeParticles`: reviewed
- L332 `ParticleChecker`: reviewed
- L337 `operator()`: reviewed
- L352 `ParticleCreator`: reviewed
- L368 `operator()`: reviewed
- L381 `createParticles`: reviewed
- L421 `ParticleChecker`: reviewed
- L430 `operator()`: reviewed
- L470 `ParticleCreator`: reported in `issues/stochastic-star-low-mass-cap-nonpositive.md`
- L487 `operator()`: reported in `issues/stochastic-star-low-mass-cap-nonpositive.md`
- L737 `createParticles`: reviewed

## `src/particles/particle_accretion.hpp`

- L31 `get_delta_rho`: reviewed
- L34 `compute_Mdot_and_r_K`: reviewed
- L181 `compute_accretion_kernel`: reviewed
- L189 `ComputeAccretionRateInBox`: reviewed
- L253 `ComputeScaleDown`: reported in `issues/sink-accretion-scaledown-stale-rate.md`
- L323 `UpdateParticleMassAndMomentumInBox`: reviewed
- L485 `UpdateParticleMassAndMomentum`: reviewed
- L509 `UpdateHydroState`: reviewed
- L537 `computeAccretion`: reviewed
- L567 `applyAccretion`: reviewed

## `src/particles/stellarpop_data.hpp`

- L30 `interpolate_whether_SN_explosion`: reviewed
- L68 `interpolate_death_time`: reviewed

## `src/particles/PhysicsParticles.hpp`

- L39 `getParticleTypeShortName`: reviewed
- L63 `getParticleSwitchName`: reviewed
- L87 `parseParticleTypeShortName`: reviewed
- L126 `PhysicsParticleDescriptorBase`: reviewed
- L147 `getMassIndex`: reviewed
- L148 `getLumIndex`: reviewed
- L149 `getBirthTimeIndex`: reviewed
- L150 `getDeathTimeIndex`: reviewed
- L151 `getAllowsCreation`: reviewed
- L152 `getAllowsDestruction`: reviewed
- L153 `getEvolutionStageIndex`: reviewed
- L154 `getAllowsAccretion`: reviewed
- L155 `getMassAtBirthIndex`: reviewed
- L156 `getForceFinestLevel`: reviewed
- L157 `getMdotIndex`: reviewed
- L158 `getAngMomIndex`: reviewed
- L161 `setForceFinestLevel`: reviewed
- L223 `depositSN`: reviewed
- L229 `computeSinkAccretion`: reviewed
- L242 `applySinkAccretion`: reviewed
- L254 `updateParticleProperties`: reviewed
- L263 `getParticleType`: reviewed
- L267 `PhysicsParticleDescriptor`: reviewed
- L286 `getParticleDataAtAllLevels`: reviewed
- L291 `getParticleDataAtLevel`: reviewed
- L297 `getNumParticles`: reviewed
- L306 `computeStellarMass`: reviewed
- L334 `computeStellarMassAtBirth`: reviewed
- L365 `computeStellarMassAtBirthBornByTime`: reviewed
- L391 `depositMass`: reviewed
- L404 `depositParticleMassDensity`: reviewed
- L434 `driftParticles`: reviewed
- L462 `kickParticles`: reviewed
- L500 `destroyParticles`: reviewed
- L508 `splitParticles`: reviewed
- L614 `computeMaxParticleSpeed`: reviewed
- L659 `depositRadiation`: reviewed
- L668 `redistribute`: reviewed
- L676 `redistribute`: reviewed
- L684 `getRealCompNames`: reviewed
- L687 `getIntCompNames`: reviewed
- L690 `writePlotFile`: reviewed
- L700 `writeCheckpoint`: reviewed
- L710 `writeUnitsFile`: reviewed
- L717 `printParticleStatistics`: reviewed
- L724 `saveParticleDataToTxtFile`: reviewed
- L732 `tagCellsAroundParticles`: reviewed
- L761 `updateParticleProperties`: reviewed
- L769 `depositSN`: reviewed
- L798 `computeSinkAccretion`: reviewed
- L806 `applySinkAccretion`: reviewed
- L814 `createParticlesFromState`: reviewed
- L833 `PhysicsParticleRegister`: reviewed
- L840 `HasMassiveParticles`: reviewed
- L851 `HasRadiatingParticles`: reviewed
- L864 `HasFormationParticles`: reviewed
- L875 `getParticleTypeName`: reviewed
- L897 `registerParticleType`: reviewed
- L933 `getParticleDescriptor`: reviewed
- L944 `depositRadiation`: reviewed
- L955 `depositMass`: reviewed
- L966 `depositParticleMassDensity`: reviewed
- L985 `depositSN`: reviewed
- L1001 `computeSinkAccretion`: reviewed
- L1013 `applySinkAccretion`: reviewed
- L1025 `redistribute`: reviewed
- L1034 `redistribute`: reviewed
- L1043 `writePlotFile`: reviewed
- L1053 `writePlotFileFiltered`: reviewed
- L1084 `saveParticleDataToTxtFileFiltered`: reported in `issues/diagparticletxt-empty-particles-skips-output.md`
- L1096 `writeCheckpoint`: reviewed
- L1106 `driftParticlesAllLevels`: reviewed
- L1119 `kickParticlesAtLevel`: reviewed
- L1131 `createParticlesFromState`: reviewed
- L1147 `destroyParticles`: reviewed
- L1160 `computeMaxParticleSpeed`: reviewed
- L1176 `refineGridsAroundParticles`: reviewed
- L1188 `updateParticleProperties`: reviewed
- L1195 `printParticleStatistics`: reviewed
- L1213 `updateSFH`: reviewed
- L1224 `computeTotalStellarMassAtBirth`: reviewed
- L1235 `computeSfrAveragedOverTime`: reviewed
- L1259 `writeSFHToMetadata`: reviewed
- L1284 `readSFH`: reviewed
- L1331 `computePhotoelectricHeatingRate`: reviewed

## `src/particles/particle_IO.hpp`

- L35 `getParticleDataAtAllLevels`: reviewed
- L164 `getParticleDataAtLevel`: reviewed
- L279 `writeUnitsFile`: reviewed
- L323 `printParticleStatistics`: reviewed
- L373 `saveParticleDataToTxtFile`: reported in `issues/particle-text-output-skips-first-int-component.md`

## `src/radiation/planck_integral.hpp`

- L29 `interpolate_planck_integral`: reviewed
- L233 `integrate_planck_from_0_to_x`: reviewed

## `src/radiation/radiation_dust_system.hpp`

- L8 `DefinePhotoelectricHeatingE1Derivative`: reviewed
- L23 `ComputeJacobianForGasAndDust`: reviewed
- L86 `ComputeJacobianForGasAndDustDecoupled`: reviewed
- L131 `ComputeJacobianForGasAndDustWithPE`: reviewed
- L199 `SolveLinearEqsWithLastColumn`: reviewed
- L230 `SolveGasDustRadiationEnergyExchange`: reported in `issues/dust-decoupled-gas-energy-jacobian-missing-cv.md`
- L599 `SolveGasDustRadiationEnergyExchangeWithPE`: reported in `issues/dust-decoupled-gas-energy-jacobian-missing-cv.md`

## `src/radiation/radiation_system.hpp`

- L133 `minmod_func`: reviewed
- L151 `MC`: reviewed
- L466 `ComputePlanckEnergyFractions`: reviewed
- L497 `ComputeNumberDensityH`: reviewed
- L503 `ComputeThermalRadiationSingleGroup`: reviewed
- L515 `ComputeThermalRadiationMultiGroup`: reviewed
- L531 `ComputeThermalRadiationTempDerivativeSingleGroup`: reviewed
- L538 `ComputeThermalRadiationTempDerivativeMultiGroup`: reviewed
- L549 `DefineBackgroundHeatingRate`: reviewed
- L556 `DefineNetCoolingRate`: reviewed
- L566 `DefineNetCoolingRateTempDerivative`: reviewed
- L574 `DefineCosmicRayHeatingRate`: reviewed
- L585 `SolveLinearEqs`: reviewed
- L593 `Solve3x3matrix`: reviewed
- L614 `SetRadEnergySource`: reviewed
- L624 `ConservedToPrimitive`: reviewed
- L651 `ComputeMaxSignalSpeed`: reviewed
- L660 `isStateValid`: reviewed
- L680 `amendRadState`: reviewed
- L716 `PredictStep`: reviewed
- L761 `AddFluxesRK2`: reviewed
- L823 `ComputeEddingtonFactor`: reviewed
- L844 `ComputeMassScalars`: reviewed
- L855 `ComputeCellOpticalDepth`: reviewed
- L924 `ComputeEddingtonTensor`: reviewed
- L970 `ComputeRadPressure`: reviewed
- L1037 `ComputeFluxes`: reviewed
- L1191 `ComputePlanckOpacity`: reviewed
- L1196 `ComputeFluxMeanOpacity`: reviewed
- L1201 `ComputeEnergyMeanOpacity`: reviewed
- L1207 `DefineOpacityExponentsAndLowerValues`: reviewed
- L1221 `ComputeRadQuantityExponents`: reviewed
- L1302 `ComputeGroupMeanOpacity`: reviewed
- L1340 `ComputeEintFromEgas`: reviewed
- L1351 `ComputeEgasFromEint`: reviewed
- L1360 `PlanckFunction`: reviewed
- L1379 `ComputeDiffusionFluxMeanOpacity`: reviewed
- L1405 `ComputeBinCenterOpacity`: reviewed
- L1418 `ComputeFluxInDiffusionLimit`: reported in `issues/diffusion-limit-flux-zero-boundary-nan.md`
- L1438 `BackwardEulerOneVariable`: reviewed
- L1471 `ComputeDustTemperatureBateKeto`: reviewed

## `src/radiation/source_terms_multi_group.hpp`

- L9 `ComputeModelDependentKappaEAndKappaP`: reviewed
- L66 `ComputeModelDependentKappaFAndDeltaTerms`: reviewed
- L106 `ComputeJacobianForGas`: reviewed
- L150 `SolveGasRadiationEnergyExchange`: reviewed
- L428 `UpdateFlux`: reviewed
- L589 `AddSourceTermsMultiGroup`: reviewed

## `src/radiation/source_terms_single_group.hpp`

- L10 `AddSourceTermsSingleGroup`: reported in `issues/single-group-radiation-beta2-zero-velocity-matrix.md`

## `src/simulation.hpp`

- L117 `formatIntVect`: reviewed
- L128 `formatRealVect`: reviewed
- L139 `YAML::as_if<T, std::optional<T>>::operator()`: reviewed
- L155 `YAML::as_if<std::string, std::optional<std::string>>::operator()`: reviewed
- L251 `builtin_BCs_fc`: reviewed
- L258 `readBCs`: reviewed
- L285 `setCustomGhostCells`: declaration
- L287 `computeMaxSignalLocal`: declaration
- L288 `printCellProperties`: declaration
- L293 `postInitialization`: reviewed
- L294 `refineGrid`: declaration
- L306 `computeBeforeTimestep`: declaration
- L307 `computeAfterTimestep`: declaration
- L308 `computeAfterEvolve`: declaration
- L309 `fillPoissonRhsAtLevel`: declaration
- L315 `ComputeDerivedVar`: declaration
- L319 `ComputeStatistics`: declaration
- L323 `FixupState`: declaration
- L327 `ErrorEst`: declaration
- L486 `needs_refinement`: reviewed
- L678 `GetParticleRegister`: reviewed
- L682 `getGitHashForQuokka`: reviewed
- L688 `getGitHashForAmrex`: reviewed
- L694 `setChkFile`: reviewed
- L696 `getOldMF_cc`: reviewed
- L698 `getNewMF_cc`: reviewed
- L700 `getOldMF_fc`: reviewed
- L705 `getNewMF_fc`: reviewed
- L710 `initialize`: reviewed
- L792 `PerformanceHints`: reviewed
- L845 `readParameters`: reviewed
- L1122 `rereadRuntimeParameters`: reviewed
- L1129 `setInitialConditions`: reviewed
- L1197 `computeTimestepAtLevel`: reviewed
- L1269 `computeTimestep`: reviewed
- L1340 `getWalltime`: reviewed
- L1347 `getCycleWalltime`: reviewed
- L1356 `evolve`: reviewed
- L1691 `calculateGpotAllLevels`: reviewed
- L1890 `gravAccelAllLevels`: reviewed
- L1905 `ellipticSolveAllLevels`: reviewed
- L1930 `setFunctorParticleAccel::operator()`: reviewed
- L1957 `kickParticlesAllLevels`: reviewed
- L2051 `particleMeshInteraction`: reviewed
- L2121 `timeStepWithSubcycling`: reviewed
- L2254 `incrementFluxRegisters`: reviewed
- L2303 `incrementEMFRegisters`: reviewed
- L2333 `getAmrInterpolaterCellCentered`: reviewed
- L2350 `getAmrInterpolaterFaceCentered`: reviewed
- L2357 `MakeNewLevelFromCoarse`: reviewed
- L2409 `RemakeLevel`: reported in `issues/remakelevel-cell-old-state-uninitialized.md`
- L2465 `ClearLevel`: reviewed
- L2485 `InterpHookNone`: reviewed
- L2492 `setBoundaryFunctor::operator()`: reviewed
- L2504 `setBoundaryFunctorFaceVar::setBoundaryFunctorFaceVar`: reviewed
- L2507 `setBoundaryFunctorFaceVar::setBoundaryFunctorFaceVar`: reviewed
- L2511 `setBoundaryFunctorFaceVar::operator()`: reviewed
- L2531 `setCustomBoundaryConditions`: declaration
- L2546 `setCustomBoundaryConditionsFaceVar`: declaration
- L2562 `setConstantDirichletBCLo`: reviewed
- L2611 `setConstantDirichletBCHi`: reviewed
- L2661 `setDiodeBCLo`: reviewed
- L2781 `setDiodeBCHi`: reviewed
- L2900 `setConstantDirichletBCFaceVarLo`: reviewed
- L2961 `setConstantDirichletBCFaceVarHi`: reviewed
- L3024 `FillPatch`: reviewed
- L3065 `setInitialConditionsAtLevel_cc`: reviewed
- L3090 `setInitialConditionsAtLevel_fc`: reviewed
- L3118 `MakeNewLevelFromScratch`: reviewed
- L3171 `fillBoundaryConditions`: reviewed
- L3260 `FillPatchWithData`: reviewed
- L3348 `FillCoarsePatch`: reviewed
- L3381 `FillCoarsePatchFaceArray`: reviewed
- L3437 `GetData`: reviewed
- L3483 `GetDataFaceArray`: reviewed
- L3515 `AverageDown`: reviewed
- L3525 `AverageDownTo`: reviewed
- L3542 `computeVolumeIntegral`: reviewed
- L3568 `InitParticles`: reviewed
- L3590 `InitPhyParticles`: reviewed
- L3699 `PlotFileName`: reviewed
- L3702 `CustomPlotFileName`: reviewed
- L3709 `AverageFCToCC`: reviewed
- L3741 `PlotFileMFAtLevel_cc`: reviewed
- L3810 `ComputeDensityFloorDebug`: reviewed
- L3849 `PlotFileMFAtLevel_fc`: reviewed
- L3871 `AverageDownDerived`: reviewed
- L3899 `PlotFileMF_cc`: reviewed
- L3918 `PlotFileMF_fc`: reviewed
- L3930 `createRuntimeDerivedFields`: reviewed
- L4012 `updateRuntimeDerivedFields`: reviewed
- L4023 `computeRuntimeDerivedVar`: reviewed
- L4034 `createDiagnostics`: reviewed
- L4110 `updateDiagnostics`: reviewed
- L4121 `doDiagnostics`: reviewed
- L4203 `GetPlotfileVarNames`: reviewed
- L4205 `GetPlotfileVarNames_fc`: reviewed
- L4219 `WritePlotFile`: reviewed
- L4292 `WriteMetadataFile`: reviewed
- L4312 `ReadMetadataFile`: reviewed
- L4340 `WriteStatisticsFile`: reviewed
- L4385 `SetLastCheckpointSymlink`: reviewed
- L4401 `WriteCheckpointFile`: reviewed
- L4518 `GotoNextLine`: reviewed
- L4525 `detectRefinementContext`: reviewed
- L4561 `readCheckpointHeader`: reviewed
- L4615 `interpolateMultiFabFromRestart`: reviewed
- L4640 `interpolateFaceMultiFabFromRestart`: reviewed
- L4764 `loadMultiFabData`: reviewed
- L4821 `loadBalanceOnRestart`: reviewed
- L4849 `ReadCheckpointFile`: reviewed
- L4907 `restartParticleContainerWithRefinement`: reviewed
- L4991 `initializeParticleContainerFromCheckpoint`: reviewed
- L5015 `writeFaceVelocitiesToDisk`: reviewed
- L5072 `writeReconstructedStatesToDisk`: reviewed

## `src/turbulence/TurbDataReader.cpp`

- L16 `read_dataset`: reviewed
- L52 `initialize_turbdata`: reviewed
- L69 `get_tabledata`: reviewed
- L90 `computeRms`: reviewed

## `src/turbulence/TurbDataReader.hpp`

- L45 `initialize_turbdata`: declaration
- L47 `read_dataset`: declaration
- L49 `get_tabledata`: declaration
- L51 `computeRms`: declaration

## `src/turbulence/TurbulentDriving.hpp`

- L36 `calculate_dispersion`: declaration
- L48 `update`: reviewed
- L59 `turbulentDriving`: reviewed
- L60 `turbulentDriving`: reviewed
- L62 `applyDriving`: reviewed
- L104 `calculate_dispersion`: reviewed

## `src/problems/Advection/testAdvection.cpp`

- L48 `ComputeExactSolution`: reviewed
- L61 `setInitialConditionsOnGrid`: reviewed
- L77 `computeReferenceSolution`: reviewed
- L127 `problem_main`: reviewed

## `src/problems/AdvectionSemiellipse/testAdvectionSemiellipse.cpp`

- L44 `ComputeExactSolution`: reviewed
- L56 `setInitialConditionsOnGrid`: reviewed
- L71 `computeReferenceSolution`: reviewed
- L120 `problem_main`: reviewed

## `src/problems/AlfvenWaveCircular/testAlfvenWaveCircular.cpp`

- L63 `computeMagneticVectorPotential_x`: reviewed
- L68 `computeMagneticVectorPotential_y`: reviewed
- L73 `computeMagneticVectorPotential_z`: reviewed
- L79 `computeWaveSolution`: reviewed
- L168 `setInitialConditionsOnGrid`: reviewed
- L190 `setInitialConditionsOnGridFaceVars`: reviewed
- L212 `computeReferenceSolution`: reviewed
- L233 `computeReferenceSolution_fc`: reviewed
- L222 `problem_main`: reviewed

## `src/problems/AlfvenWaveLinear/testAlfvenWaveLinear.cpp`

- L60 `computeMagnitude`: reviewed
- L65 `computeDotProduct`: reviewed
- L70 `computeCrossProduct`: reviewed
- L77 `normalizeVector`: reviewed
- L152 `rotatePRF2MRF`: reviewed
- L164 `rotateMRF2PRF`: reviewed
- L176 `computeVectorPotentialComponent_prf`: reviewed
- L200 `Ax_prf`: reviewed
- L205 `Ay_prf`: reviewed
- L210 `Az_prf`: reviewed
- L216 `computeWaveSolution`: reviewed
- L295 `setInitialConditionsOnGrid`: reviewed
- L315 `setInitialConditionsOnGridFaceVars`: reviewed
- L334 `computeReferenceSolution`: reviewed
- L354 `computeReferenceSolution_fc`: reviewed
- L370 `problem_main`: reviewed

## `src/problems/AlfvenWaveLinearConvergence/testAlfvenWaveLinearConvergence.cpp`

- L64 `computeMagnitude`: reviewed
- L69 `computeDotProduct`: reviewed
- L74 `computeCrossProduct`: reviewed
- L81 `normalizeVector`: reviewed
- L156 `rotatePRF2MRF`: reviewed
- L168 `rotateMRF2PRF`: reviewed
- L180 `computeVectorPotentialComponent_prf`: reviewed
- L204 `Ax_prf`: reviewed
- L209 `Ay_prf`: reviewed
- L214 `Az_prf`: reviewed
- L220 `computeWaveSolution`: reviewed
- L297 `setInitialConditionsOnGrid`: reviewed
- L317 `setInitialConditionsOnGridFaceVars`: reviewed
- L336 `computeReferenceSolution`: reviewed
- L356 `computeReferenceSolution_fc`: reviewed
- L372 `runWaveTest`: reviewed
- L479 `problem_main`: reviewed

## `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp`

- L71 `setInitialConditionsOnGrid`: reviewed
- L85 `createInitialCICParticles`: reviewed
- L102 `ComputeDerivedVar`: reviewed
- L113 `computeAfterTimestep`: reviewed
- L157 `problem_main`: reported in `issues/binary-orbit-cic-deviation-failure-status.md`

## `src/problems/BrioWuShockTube/testBrioWuShockTube.cpp`

- L61 `setInitialConditionsOnGrid`: reviewed
- L116 `setInitialConditionsOnGridFaceVars`: reviewed
- L146 `setCustomBoundaryConditions`: reviewed
- L190 `setCustomBoundaryConditionsFaceVar`: reviewed
- L209 `refineGrid`: reviewed
- L235 `problem_main`: reviewed

## `src/problems/CurrentSheet/testCurrentSheet.cpp`

- L55 `setInitialConditionsOnGrid`: reviewed
- L80 `setInitialConditionsOnGridFaceVars`: reviewed
- L109 `problem_main`: reviewed

## `src/problems/DiskGalaxy/testDiskGalaxy.cpp`

- L104 `preCalculateInitialConditions`: reviewed
- L170 `setInitialConditionsOnGrid`: reviewed
- L455 `setInitialConditionsOnGridFaceVars`: reviewed
- L508 `createInitialCICParticles`: reviewed
- L522 `refineGrid`: reviewed
- L567 `ComputeDerivedVar`: reviewed
- L714 `ComputeStatistics`: reviewed
- L899 `problem_main`: reviewed

## `src/problems/DustAdvection/testDustAdvection.cpp`

- L49 `setInitialConditionsOnGrid`: reviewed
- L91 `computeReferenceSolution`: reviewed
- L231 `problem_main`: reviewed

## `src/problems/DustAdvection3D/testDustAdvection3D.cpp`

- L52 `setInitialConditionsOnGrid`: reviewed
- L95 `computeReferenceSolution`: reviewed
- L344 `problem_main`: reviewed

## `src/problems/DustDamping/testDustDamping.cpp`

- L113 `computeDustStoppingTime`: reviewed
- L123 `setInitialConditionsOnGrid`: reviewed
- L160 `computeAfterTimestep`: reviewed
- L196 `analytic_velocity`: reviewed
- L198 `v_gas_analytic`: reviewed
- L200 `v_dust1_analytic`: reviewed
- L202 `v_dust2_analytic`: reviewed
- L205 `E_gas_analytic`: reviewed
- L236 `problem_main`: reviewed

## `src/problems/DustDampingIteration/testDustDampingIteration.cpp`

- L102 `computeDustStoppingTime` (`DustDampingWithCorrection`): reviewed
- L111 `computeDustStoppingTime` (`DustDampingWithoutCorrection`): reviewed
- L119 `setInitialConditionsOnGrid` (`DustDampingWithCorrection`): reviewed
- L156 `setInitialConditionsOnGrid` (`DustDampingWithoutCorrection`): reviewed
- L193 `computeAfterTimestep` (`DustDampingWithCorrection`): reviewed
- L228 `computeAfterTimestep` (`DustDampingWithoutCorrection`): reviewed
- L263 `run_reference_simulation`: reviewed
- L306 `run_iterative_with_correction`: reviewed
- L348 `run_iterative_without_correction`: reviewed
- L390 `compute_relative_error`: reviewed
- L433 `problem_main`: reviewed

## `src/problems/DustyShock/testDustyShock.cpp`

- L55 `computeDustStoppingTime`: reviewed
- L66 `setInitialConditionsOnGrid`: reviewed
- L107 `solve_quadratic_root_in_0_1`: reviewed
- L135 `linear_interpolate`: reviewed
- L156 `problem_main`: reviewed

## `src/problems/DustDampingWithExternalForce/testDustDampingWithExternalForce.cpp`

- L92 `ComputeReciprocalStoppingTime`: reviewed
- L103 `setInitialConditionsOnGrid`: reviewed
- L140 `computeAfterTimestep`: reviewed
- L175 `v_gas_analytic`: reviewed
- L177 `v_dust1_analytic`: reviewed
- L182 `v_dust2_analytic`: reviewed
- L187 `E_gas_analytic`: reviewed
- L223 `addStrangSplitSources`: reviewed
- L257 `problem_main`: reviewed

## `src/problems/DustSoundwave/testDustSoundwave.cpp`

- L33 `real_part_analytic`: reviewed
- L43 `v_gas_analytic`: reviewed
- L45 `rho_gas_analytic`: reviewed
- L47 `v_dust_analytic`: reviewed
- L49 `rho_dust_analytic`: reviewed
- L86 `ComputeReciprocalStoppingTime`: reviewed
- L95 `setInitialConditionsOnGrid`: reviewed
- L158 `computeAfterTimestep`: reviewed
- L178 `problem_main`: reviewed

## `src/problems/EntropyWaveConvergence/testEntropyWaveConvergence.cpp`

- L68 `computeMagnitude`: reviewed
- L73 `computeDotProduct`: reviewed
- L78 `computeCrossProduct`: reviewed
- L85 `normalizeVector`: reviewed
- L112 `rotatePRF2MRF`: reviewed
- L119 `rotateMRF2PRF`: reviewed
- L127 `computeVectorPotentialComponent_prf`: reviewed
- L144 `Ax_prf`: reviewed
- L150 `Ay_prf`: reviewed
- L156 `Az_prf`: reviewed
- L163 `computeWaveSolution`: reviewed
- L234 `setInitialConditionsOnGrid`: reviewed
- L252 `setInitialConditionsOnGridFaceVars`: reviewed
- L273 `computeReferenceSolution`: reviewed
- L291 `computeReferenceSolution_fc`: reviewed
- L308 `runWaveTest`: reviewed
- L414 `problem_main`: reviewed

## `src/problems/FCQuantities/testFCQuantities.cpp`

- L59 `computeWaveSolution`: reviewed
- L83 `setInitialConditionsOnGrid`: reviewed
- L101 `setInitialConditionsOnGridFaceVars`: reviewed
- L137 `setAmrNCell`: reviewed
- L143 `setPlotfileParams`: reviewed
- L151 `checkDivFreeRestart`: reviewed
- L184 `problem_main`: reviewed

## `src/problems/FastWave/testFastWave.cpp`

- L78 `computeMagneticVectorPotential_x`: reviewed
- L82 `computeMagneticVectorPotential_y`: reviewed
- L86 `computeMagneticVectorPotential_z`: reviewed
- L89 `computeWaveSolution`: reviewed
- L158 `setInitialConditionsOnGrid`: reviewed
- L178 `setInitialConditionsOnGridFaceVars`: reviewed
- L199 `computeReferenceSolution`: reviewed
- L218 `computeReferenceSolution_fc`: reviewed
- L236 `problem_main`: reviewed

## `src/problems/FastWaveConvergence/testFastWaveConvergence.cpp`

- L58 `computeMagnitude`: reviewed
- L63 `computeDotProduct`: reviewed
- L68 `computeCrossProduct`: reviewed
- L75 `normalizeVector`: reviewed
- L150 `rotatePRF2MRF`: reviewed
- L162 `rotateMRF2PRF`: reviewed
- L170 `computeVectorPotentialComponent_prf`: reviewed
- L219 `Ax_prf`: reviewed
- L224 `Ay_prf`: reviewed
- L229 `Az_prf`: reviewed
- L235 `computeWaveSolution`: reviewed
- L350 `setInitialConditionsOnGrid`: reviewed
- L368 `setInitialConditionsOnGridFaceVars`: reviewed
- L387 `computeReferenceSolution`: reviewed
- L405 `computeReferenceSolution_fc`: reviewed
- L423 `runWaveTest`: reviewed
- L532 `problem_main`: reviewed

## `src/problems/FieldLoop/testFieldLoop.cpp`

- L28 `amrex_get_enum_traits`: reviewed
- L52 `setInitialConditionsOnGrid`: reported in `issues/field-loop-z-kinetic-energy-omitted.md`
- L96 `setInitialConditionsOnGridFaceVars`: reviewed
- L128 `refineGrid`: reviewed
- L172 `ComputeDerivedVar`: reviewed
- L204 `problem_main`: reviewed

## `src/problems/GravRadParticle3D/testGravRadParticle3D.cpp`

- L70 `createInitialCICRadParticles`: reviewed
- L78 `createInitialCICParticles`: reviewed
- L86 `createInitialRadParticles`: reviewed
- L98 `ComputePlanckOpacity`: reviewed
- L103 `ComputeFluxMeanOpacity`: reviewed
- L108 `setInitialConditionsOnGrid`: reviewed
- L135 `checkGasDensityProjection`: reviewed
- L185 `problem_main`: reviewed

## `src/problems/HydroBlast3D/testHydroBlast3D.cpp`

- L57 `preCalculateInitialConditions`: reviewed
- L64 `setInitialConditionsOnGrid`: reviewed
- L107 `refineGrid`: reviewed
- L142 `computeAfterEvolve`: reviewed
- L220 `problem_main`: reviewed

## `src/problems/HydroContact/testHydroContact.cpp`

- L47 `setInitialConditionsOnGrid`: reviewed
- L96 `computeReferenceSolution`: reviewed
- L192 `problem_main`: reviewed

## `src/problems/HydroHighMach/testHydroHighMach.cpp`

- L52 `setInitialConditionsOnGrid`: reviewed
- L95 `computeReferenceSolution`: reviewed
- L255 `problem_main`: reviewed

## `src/problems/HydroLeblanc/testHydroLeblanc.cpp`

- L52 `setInitialConditionsOnGrid`: reviewed
- L103 `setCustomBoundaryConditions`: reviewed
- L165 `computeReferenceSolution`: reviewed
- L344 `problem_main`: reviewed

## `src/problems/HydroQuirk/testHydroQuirk.cpp`

- L67 `setInitialConditionsOnGrid`: reviewed
- L130 `getDeltaEntropyVector`: reviewed
- L136 `computeAfterTimestep`: reviewed
- L189 `computeAfterEvolve`: reviewed
- L205 `setCustomBoundaryConditions`: reviewed
- L239 `problem_main`: reviewed

## `src/problems/HydroSMS/testHydroSMS.cpp`

- L48 `setInitialConditionsOnGrid`: reviewed
- L93 `setCustomBoundaryConditions`: reviewed
- L151 `computeReferenceSolution`: reviewed
- L263 `problem_main`: reviewed

## `src/problems/HydroShocktube/testHydroShocktube.cpp`

- L58 `setInitialConditionsOnGrid`: reviewed
- L109 `setCustomBoundaryConditions`: reviewed
- L146 `refineGrid`: reviewed
- L174 `computeReferenceSolution`: reviewed
- L340 `problem_main`: reviewed

## `src/problems/HydroShocktubeCMA/testHydroShocktubeCMA.cpp`

- L69 `setInitialConditionsOnGrid`: reviewed
- L124 `setCustomBoundaryConditions`: reported in `issues/hydro-shocktube-cma-right-scalar2-not-density.md`
- L179 `refineGrid`: reviewed
- L205 `computeAfterTimestep`: reviewed
- L238 `problem_main`: reviewed

## `src/problems/HydroShuOsher/testHydroShuOsher.cpp`

- L45 `setInitialConditionsOnGrid`: reviewed
- L92 `setCustomBoundaryConditions`: reviewed
- L145 `computeReferenceSolution`: reviewed
- L274 `problem_main`: reported in `issues/hydro-shuosher-bc-component-zero-only.md`

## `src/problems/HydroVacuum/testHydroVacuum.cpp`

- L48 `setInitialConditionsOnGrid`: reviewed
- L94 `setCustomBoundaryConditions`: reviewed
- L150 `computeReferenceSolution`: reviewed
- L306 `problem_main`: reviewed

## `src/problems/HydroWave/testHydroWave.cpp`

- L54 `computeWaveSolution`: reviewed
- L81 `setInitialConditionsOnGrid`: reviewed
- L95 `problem_main`: reviewed

## `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp`

- L52 `computeWaveSolution`: reviewed
- L79 `setInitialConditionsOnGrid`: reviewed
- L93 `runWaveTest`: reviewed
- L166 `problem_main`: reviewed

## `src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp`

- L52 `setCustomBoundaryConditions`: reported in `issues/hydrostatic-bc-geometry-raw-pointers.md`
- L75 `setInitialConditionsOnGrid`: reviewed
- L105 `computeReferenceSolution`: reviewed
- L182 `problem_main`: reviewed

## `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp`

- L64 `computeRadiusSq`: reviewed
- L74 `computeRadialProfile`: reviewed
- L76 `Az`: reviewed
- L89 `computeVortexSolution`: reviewed
- L126 `setInitialConditionsOnGrid`: reviewed
- L141 `setInitialConditionsOnGridFaceVars`: reviewed
- L165 `computeReferenceSolution`: reviewed
- L194 `computeReferenceSolution_fc`: reviewed
- L225 `problem_main`: reviewed

## `src/problems/MHDBitwiseICs/testMHDBitwiseICs.cpp`

- L53 `computeWaveSolution`: reviewed
- L79 `setInitialConditionsOnGrid`: reviewed
- L100 `setInitialConditionsOnGridFaceVars`: reviewed
- L125 `computeReferenceSolution`: reviewed
- L143 `computeReferenceSolution_fc`: reviewed
- L156 `verifyPeriodicBCs`: reviewed
- L246 `problem_main`: reviewed

## `src/problems/MHDBlast/testMHDBlast.cpp`

- L44 `setInitialConditionsOnGrid`: reviewed
- L78 `setInitialConditionsOnGridFaceVars`: reviewed
- L99 `refineGrid`: reviewed
- L134 `ComputeDerivedVar`: reviewed
- L166 `problem_main`: reviewed

## `src/problems/MHDQuirk/testMHDQuirk.cpp`

- L71 `setInitialConditionsOnGridFaceVars`: reviewed
- L86 `setInitialConditionsOnGrid`: reviewed
- L145 `getDeltaEntropyVector`: reviewed
- L151 `computeAfterTimestep`: reviewed
- L206 `computeAfterEvolve`: reviewed
- L225 `setCustomBoundaryConditions`: reviewed
- L256 `problem_main`: reviewed

## `src/problems/NscbcChannel/testNscbcChannel.cpp`

- L74 `setInitialConditionsOnGrid`: reviewed
- L101 `setCustomBoundaryConditions`: reviewed
- L126 `problem_main`: reviewed

## `src/problems/NscbcVortex/testNscbcVortex.cpp`

- L80 `setInitialConditionsOnGrid`: reviewed
- L126 `setCustomBoundaryConditions`: reviewed
- L155 `problem_main`: reviewed

## `src/problems/ODEIntegration/testODEIntegration.cpp`

- L39 `cooling_function`: reviewed
- L55 `ODECoolingFunctor::ODECoolingFunctor`: reviewed
- L57 `ODECoolingFunctor::operator()`: reviewed
- L66 `problem_main`: reviewed

## `src/problems/OrszagTang/testOrszagTang.cpp`

- L51 `A_z`: reviewed
- L56 `B_x`: reviewed
- L61 `B_y`: reviewed
- L66 `setInitialConditionsOnGrid`: reviewed
- L100 `setInitialConditionsOnGridFaceVars`: reviewed
- L123 `problem_main`: reviewed

## `src/problems/ParticleAccretion/testParticleAccretion.cpp`

- L205 `createInitialSinkParticles`: reviewed
- L290 `setInitialConditionsOnGrid`: reviewed
- L360 `setInitialConditionsOnGridFaceVars`: reviewed
- L379 `refineGrid`: reviewed
- L401 `computeAfterTimestep`: reviewed
- L422 `problem_main`: reviewed

## `src/problems/ParticleCreation/testParticleCreation.cpp`

- L71 `createInitialTestParticles`: reviewed
- L123 `ParticleChecker::ParticleChecker`: reviewed
- L125 `ParticleChecker::operator()`: reviewed
- L170 `ParticleCreator::ParticleCreator`: reviewed
- L172 `ParticleCreator::operator()`: reviewed
- L221 `ParticleCreationTraits<ParticleType::Test>::createParticles`: reviewed
- L241 `setInitialConditionsOnGrid`: reviewed
- L250 `setInitialConditionsOnGridFaceVars`: reviewed
- L260 `problem_main`: reviewed

## `src/problems/ParticleDeposition/testParticleDeposition.cpp`

- L71 `setInitialConditionsOnGrid`: reviewed
- L86 `createInitialCICParticles`: reviewed
- L92 `createInitialTestParticles`: reviewed
- L127 `computeAfterTimestep`: reviewed
- L156 `ComputeDerivedVar`: reviewed
- L166 `problem_main`: reviewed

## `src/problems/ParticleRadiation/testParticleRadiation.cpp`

- L82 `DefineOpacityExponentsAndLowerValues`: reviewed
- L84 `createInitialStochasticStellarPopParticles`: reviewed
- L131 `setInitialConditionsOnGrid`: reviewed
- L153 `problem_main`: reviewed

## `src/problems/ParticleSF/testParticleSF.cpp`

- L60 `setInitialConditionsOnGrid`: reviewed
- L83 `refineGrid`: reviewed
- L95 `computeAfterTimestep`: reported in `issues/particle-sf-signed-relative-errors-pass-underproduction.md`
- L231 `problem_main`: reported in `issues/particle-sf-signed-relative-errors-pass-underproduction.md`

## `src/problems/ParticleSink/testParticleSink.cpp`

- L58 `createInitialSinkParticles`: reviewed
- L95 `setInitialConditionsOnGrid`: reviewed
- L119 `setInitialConditionsOnGridFaceVars`: reviewed
- L140 `refineGrid`: reviewed
- L164 `problem_main`: reported in `issues/particle-sink-boost-vel-nan-check.md`

## `src/problems/ParticleSinkFormation/testParticleSinkFormation.cpp`

- L70 `setInitialConditionsOnGrid`: reviewed
- L101 `setInitialConditionsOnGridFaceVars`: reviewed
- L112 `refineGrid`: reviewed
- L128 `problem_main`: reviewed

## `src/problems/PassiveScalar/testPassiveScalar.cpp`

- L67 `setInitialConditionsOnGrid`: reviewed
- L129 `computeReferenceSolution`: reviewed
- L222 `refineGrid`: reviewed
- L249 `problem_main`: reviewed

## `src/problems/PopIII/testPopIII.cpp`

- L91 `preCalculateInitialConditions`: reviewed
- L219 `setInitialConditionsOnGrid`: reviewed
- L332 `refineGrid`: reviewed
- L368 `ComputeDerivedVar`: reviewed
- L428 `problem_main`: reviewed

## `src/problems/PrimordialChem/testPrimordialChem.cpp`

- L80 `preCalculateInitialConditions`: reviewed
- L135 `setInitialConditionsOnGrid`: reviewed
- L249 `problem_main`: reviewed

## `src/problems/RadDust/testRadDust.cpp`

- L84 `ComputePlanckOpacity`: reviewed
- L89 `ComputeFluxMeanOpacity`: reviewed
- L94 `ComputeThermalRadiationSingleGroup`: reviewed
- L112 `ComputeThermalRadiationTempDerivativeSingleGroup`: reviewed
- L120 `setInitialConditionsOnGrid`: reviewed
- L129 `computeAfterTimestep`: reviewed
- L149 `problem_main`: reviewed

## `src/problems/RadDustMG/testRadDustMG.cpp`

- L84 `DefineOpacityExponentsAndLowerValues`: reviewed
- L116 `ComputeThermalRadiationMultiGroup`: reviewed
- L121 `ComputeThermalRadiationTempDerivativeMultiGroup`: reviewed
- L129 `setInitialConditionsOnGrid`: reviewed
- L150 `computeAfterTimestep`: reviewed
- L173 `problem_main`: reviewed

## `src/problems/RadForce/testRadForce.cpp`

- L82 `ComputePlanckOpacity`: reviewed
- L84 `ComputeFluxMeanOpacity`: reviewed
- L89 `setInitialConditionsOnGrid`: reviewed
- L135 `setCustomBoundaryConditions`: reviewed
- L164 `problem_main`: reviewed

## `src/problems/RadLineCooling/testRadLineCooling.cpp`

- L99 `DefineNetCoolingRate`: reviewed
- L105 `DefineNetCoolingRateTempDerivative`: reviewed
- L111 `DefineCosmicRayHeatingRate`: reviewed
- L124 `ComputePlanckOpacity`: reviewed
- L129 `ComputeFluxMeanOpacity`: reviewed
- L134 `DefineOpacityExponentsAndLowerValues`: reviewed
- L136 `setInitialConditionsOnGrid`: reviewed
- L172 `computeAfterTimestep`: reviewed
- L191 `problem_main`: reviewed

## `src/problems/RadLineCoolingMG/testRadLineCoolingMG.cpp`

- L99 `DefinePhotoelectricHeatingE1Derivative`: reviewed
- L106 `DefineNetCoolingRate`: reviewed
- L116 `DefineNetCoolingRateTempDerivative`: reviewed
- L125 `DefineCosmicRayHeatingRate`: reviewed
- L132 `DefineOpacityExponentsAndLowerValues`: reviewed
- L145 `setInitialConditionsOnGrid`: reviewed
- L171 `computeAfterTimestep`: reviewed
- L190 `problem_main`: reviewed

## `src/problems/RadMarshak/testRadMarshak.cpp`

- L71 `ComputePlanckOpacity`: reviewed
- L76 `ComputeFluxMeanOpacity`: reviewed
- L84 `ComputeTgasFromEint`: reviewed
- L92 `ComputeEintFromTgas`: reviewed
- L100 `ComputeEintTempDerivative`: reviewed
- L117 `setCustomBoundaryConditions`: reviewed
- L164 `setInitialConditionsOnGrid`: reviewed
- L187 `problem_main`: reviewed

## `src/problems/RadMarshakAsymptotic/testRadMarshakAsymptotic.cpp`

- L62 `ComputePlanckOpacity`: reviewed
- L69 `ComputeFluxMeanOpacity`: reviewed
- L74 `ComputeEddingtonFactor`: reviewed
- L81 `setCustomBoundaryConditions`: reviewed
- L144 `setInitialConditionsOnGrid`: reviewed
- L167 `problem_main`: reviewed

## `src/problems/RadMarshakCGS/testRadMarshakCGS.cpp`

- L69 `ComputePlanckOpacity`: reviewed
- L74 `ComputeFluxMeanOpacity`: reviewed
- L82 `ComputeTgasFromEint`: reviewed
- L90 `ComputeEintFromTgas`: reviewed
- L98 `ComputeEintTempDerivative`: reviewed
- L115 `setCustomBoundaryConditions`: reviewed
- L173 `setInitialConditionsOnGrid`: reviewed
- L197 `problem_main`: reviewed

## `src/problems/RadMarshakDust/testRadMarshakDust.cpp`

- L90 `ComputePlanckOpacity`: reviewed
- L95 `ComputeFluxMeanOpacity`: reviewed
- L102 `DefineOpacityExponentsAndLowerValues`: reviewed
- L117 `setInitialConditionsOnGrid`: reviewed
- L144 `setCustomBoundaryConditions`: reviewed
- L178 `problem_main`: reviewed

## `src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp`

- L89 `DefinePhotoelectricHeatingE1Derivative`: reviewed
- L104 `DefineOpacityExponentsAndLowerValues`: reviewed
- L119 `setInitialConditionsOnGrid`: reviewed
- L145 `setCustomBoundaryConditions`: reviewed
- L182 `problem_main`: reviewed

## `src/problems/RadMarshakVaytet/testRadMarshakVaytet.cpp`

- L135 `DefineOpacityExponentsAndLowerValues`: reviewed
- L175 `setCustomBoundaryConditions`: reviewed
- L221 `setInitialConditionsOnGrid`: reviewed
- L249 `problem_main`: reviewed

## `src/problems/RadMatterCoupling/testRadMatterCoupling.cpp`

- L63 `ComputePlanckOpacity`: reviewed
- L68 `ComputeFluxMeanOpacity`: reviewed
- L76 `ComputeTgasFromEint`: reviewed
- L84 `ComputeEintFromTgas`: reviewed
- L92 `ComputeEintTempDerivative`: reviewed
- L110 `setInitialConditionsOnGrid`: reviewed
- L130 `computeAfterTimestep`: reviewed
- L151 `problem_main`: reviewed

## `src/problems/RadMatterCouplingRSLA/testRadMatterCouplingRSLA.cpp`

- L66 `ComputePlanckOpacity`: reviewed
- L71 `ComputeFluxMeanOpacity`: reviewed
- L79 `ComputeTgasFromEint`: reviewed
- L87 `ComputeEintFromTgas`: reviewed
- L95 `ComputeEintTempDerivative`: reviewed
- L113 `setInitialConditionsOnGrid`: reviewed
- L133 `computeAfterTimestep`: reviewed
- L154 `problem_main`: reviewed

## `src/problems/RadStreaming/testRadStreaming.cpp`

- L62 `ComputePlanckOpacity`: reviewed
- L67 `ComputeFluxMeanOpacity`: reviewed
- L72 `setInitialConditionsOnGrid`: reviewed
- L106 `setCustomBoundaryConditions`: reviewed
- L157 `problem_main`: reviewed

## `src/problems/RadStreamingY/testRadStreamingY.cpp`

- L62 `ComputePlanckOpacity`: reviewed
- L67 `ComputeFluxMeanOpacity`: reviewed
- L72 `setInitialConditionsOnGrid`: reviewed
- L97 `setCustomBoundaryConditions`: reviewed
- L143 `problem_main`: reviewed

## `src/problems/RadSuOlson/testRadSuOlson.cpp`

- L85 `ComputePlanckOpacity`: reviewed
- L90 `ComputeFluxMeanOpacity`: reviewed
- L98 `ComputeTgasFromEint`: reviewed
- L106 `ComputeEintFromTgas`: reviewed
- L114 `ComputeEintTempDerivative`: reviewed
- L132 `SetRadEnergySource`: reviewed
- L158 `setInitialConditionsOnGrid`: reviewed
- L181 `problem_main`: reviewed

## `src/problems/RadTube/testRadTube.cpp`

- L83 `DefineOpacityExponentsAndLowerValues`: reviewed
- L101 `preCalculateInitialConditions`: reviewed
- L145 `setInitialConditionsOnGrid`: reviewed
- L193 `setCustomBoundaryConditions`: reviewed
- L251 `problem_main`: reviewed

## `src/problems/RadhydroBB/testRadhydroBB.cpp`

- L158 `compute_exact_bb`: reviewed
- L166 `setInitialConditionsOnGrid`: reviewed
- L191 `problem_main`: reviewed

## `src/problems/RadhydroPulse/testRadhydroPulse.cpp`

- L92 `compute_initial_Tgas`: reviewed
- L100 `compute_exact_rho`: reviewed
- L107 `ComputePlanckOpacity`: reviewed
- L111 `ComputePlanckOpacity`: reviewed
- L116 `ComputeFluxMeanOpacity`: reviewed
- L120 `ComputeFluxMeanOpacity`: reviewed
- L125 `setInitialConditionsOnGrid`: reviewed
- L156 `setInitialConditionsOnGrid`: reviewed
- L197 `problem_main`: reviewed

## `src/problems/RadhydroPulseDyn/testRadhydroPulseDyn.cpp`

- L92 `compute_initial_Tgas`: reviewed
- L100 `compute_exact_rho`: reviewed
- L107 `ComputePlanckOpacity`: reviewed
- L111 `ComputePlanckOpacity`: reviewed
- L116 `ComputeFluxMeanOpacity`: reviewed
- L120 `ComputeFluxMeanOpacity`: reviewed
- L125 `setInitialConditionsOnGrid`: reviewed
- L156 `setInitialConditionsOnGrid`: reviewed
- L197 `problem_main`: reviewed

## `src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp`

- L93 `compute_initial_Tgas`: reviewed
- L101 `compute_exact_rho`: reviewed
- L108 `ComputePlanckOpacity`: reviewed
- L113 `ComputePlanckOpacity`: reviewed
- L119 `ComputeFluxMeanOpacity`: reviewed
- L124 `ComputeFluxMeanOpacity`: reviewed
- L130 `setInitialConditionsOnGrid`: reviewed
- L161 `setInitialConditionsOnGrid`: reviewed
- L194 `problem_main`: reviewed

## `src/problems/RadhydroPulseMGconst/testRadhydroPulseMGconst.cpp`

- L76 `compute_initial_Tgas`: reviewed
- L84 `compute_exact_rho`: reviewed
- L117 `ComputePlanckOpacity`: reviewed
- L119 `ComputeFluxMeanOpacity`: reviewed
- L124 `setInitialConditionsOnGrid`: reviewed
- L190 `DefineOpacityExponentsAndLowerValues`: reviewed
- L203 `setInitialConditionsOnGrid`: reviewed
- L241 `problem_main`: reviewed

## `src/problems/RadhydroPulseMGint/testRadhydroPulseMGint.cpp`

- L153 `compute_initial_Tgas`: reviewed
- L161 `compute_exact_rho`: reviewed
- L169 `compute_kappa`: reviewed
- L179 `DefineOpacityExponentsAndLowerValues`: reviewed
- L212 `ComputePlanckOpacity`: reviewed
- L218 `ComputeFluxMeanOpacity`: reviewed
- L224 `setInitialConditionsOnGrid`: reviewed
- L265 `setInitialConditionsOnGrid`: reviewed
- L299 `problem_main`: reviewed

## `src/problems/RadhydroShell/testRadhydroShell.cpp`

- L90 `SetRadEnergySource`: reviewed
- L121 `ComputePlanckOpacity`: reviewed
- L127 `ComputeFluxMeanOpacity`: reviewed
- L143 `preCalculateInitialConditions`: reviewed
- L181 `setInitialConditionsOnGrid`: reviewed
- L255 `refineGrid`: reviewed
- L293 `problem_main`: reviewed

## `src/problems/RadhydroShock/testRadhydroShock.cpp`

- L97 `ComputePlanckOpacity`: reviewed
- L102 `ComputeFluxMeanOpacity`: reviewed
- L107 `ComputeEddingtonFactor`: reviewed
- L114 `setCustomBoundaryConditions`: reviewed
- L164 `setInitialConditionsOnGrid`: reviewed
- L215 `problem_main`: reviewed

## `src/problems/RadhydroShockCGS/testRadhydroShockCGS.cpp`

- L98 `ComputePlanckOpacity`: reviewed
- L103 `ComputeFluxMeanOpacity`: reviewed
- L108 `ComputeEddingtonFactor`: reviewed
- L115 `setCustomBoundaryConditions`: reviewed
- L172 `setInitialConditionsOnGrid`: reviewed
- L232 `problem_main`: reviewed

## `src/problems/RadhydroShockMultigroup/testRadhydroShockMultigroup.cpp`

- L90 `DefineOpacityExponentsAndLowerValues`: reviewed
- L101 `ComputeEddingtonFactor`: reviewed
- L108 `setCustomBoundaryConditions`: reviewed
- L170 `setInitialConditionsOnGrid`: reviewed
- L222 `problem_main`: reviewed

## `src/problems/RadhydroUniformAdvecting/testRadhydroUniformAdvecting.cpp`

- L90 `ComputePlanckOpacity`: reviewed
- L95 `ComputeFluxMeanOpacity`: reviewed
- L100 `setInitialConditionsOnGrid`: reviewed
- L139 `problem_main`: reviewed

## `src/problems/RandomBlast/testRandomBlast.cpp`

- L64 `setInitialConditionsOnGrid`: reviewed
- L96 `createInitialStochasticStellarPopParticles`: reviewed
- L132 `computeAfterTimestep`: reviewed
- L138 `ComputeDerivedVar`: reviewed
- L166 `problem_main`: reviewed

## `src/problems/RayleighTaylor3D/testRayleighTaylor3D.cpp`

- L57 `setInitialConditionsOnGrid`: reviewed
- L109 `addStrangSplitSources`: reviewed
- L140 `refineGrid`: reviewed
- L173 `computeAfterTimestep`: reviewed
- L200 `problem_main`: reviewed

## `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp`

- L35 `readReferenceCSV`: reviewed
- L109 `setInitialConditionsOnGrid`: reviewed
- L136 `computeAfterTimestep`: reviewed
- L164 `problem_main`: reviewed

## `src/problems/SN/testSN.cpp`

- L84 `createInitialTestParticles`: reviewed
- L116 `setInitialConditionsOnGrid`: reviewed
- L174 `setInitialConditionsOnGridFaceVars`: reviewed
- L184 `refineGrid`: reviewed
- L208 `computeAfterTimestep`: reviewed
- L217 `problem_main`: reported in `issues/sn-validation-hidden-behind-python.md`

## `src/problems/ShockCloud/testShockCloud.cpp`

- L86 `setInitialConditionsOnGrid`: reviewed
- L148 `setCustomBoundaryConditions`: reviewed
- L207 `computeAfterTimestep`: reviewed
- L275 `ComputeDerivedVar`: reviewed
- L483 `ComputeCellTempResampled`: reviewed
- L495 `ComputeStatistics`: reviewed
- L658 `refineGrid`: reviewed
- L688 `problem_main`: reviewed

## `src/problems/SlowWaveConvergence/testSlowWaveConvergence.cpp`

- L58 `computeMagnitude`: reviewed
- L63 `computeDotProduct`: reviewed
- L68 `computeCrossProduct`: reviewed
- L75 `normalizeVector`: reviewed
- L150 `rotatePRF2MRF`: reviewed
- L162 `rotateMRF2PRF`: reviewed
- L170 `computeVectorPotentialComponent_prf`: reviewed
- L220 `Ax_prf`: reviewed
- L225 `Ay_prf`: reviewed
- L230 `Az_prf`: reviewed
- L236 `computeWaveSolution`: reviewed
- L359 `setInitialConditionsOnGrid`: reviewed
- L377 `setInitialConditionsOnGridFaceVars`: reviewed
- L396 `computeReferenceSolution`: reviewed
- L414 `computeReferenceSolution_fc`: reviewed
- L432 `runWaveTest`: reviewed
- L541 `problem_main`: reviewed

## `src/problems/SphericalCollapse/testSphericalCollapse.cpp`

- L59 `setInitialConditionsOnGrid`: reviewed
- L97 `createInitialCICParticles`: reviewed
- L110 `refineGrid`: reviewed
- L130 `ComputeDerivedVar`: reviewed
- L141 `problem_main`: reviewed

## `src/problems/StarCluster/testStarCluster.cpp`

- L81 `preCalculateInitialConditions`: reviewed
- L125 `setInitialConditionsOnGrid`: reviewed
- L175 `refineGrid`: reviewed
- L196 `ComputeDerivedVar`: reviewed
- L211 `problem_main`: reviewed

## `src/problems/TallBoxSf/testTallBoxSf.cpp`

- L86 `createInitialStochasticStellarPopParticles`: reviewed
- L125 `refineGrid`: reviewed
- L154 `preCalculateInitialConditions`: reviewed
- L209 `setInitialConditionsOnGrid`: reviewed
- L300 `ComputeDerivedVar`: reviewed
- L375 `addStrangSplitSources`: reviewed
- L449 `setCustomBoundaryConditions`: reviewed
- L459 `problem_main`: reviewed

## `src/problems/Turbulence/testTurbulence.cpp`

- L53 `setInitialConditionsOnGrid`: reviewed
- L70 `refineGrid`: reviewed
- L103 `computeAfterTimestep`: reviewed
- L114 `problem_main`: reviewed
