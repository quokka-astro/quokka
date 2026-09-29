# Additional registered ParmParse options

The [runtime parameter tables](parameters.md) give types and defaults for the main simulation controls. The options below are also registered in Quokka and are read by individual problems, diagnostics, or supporting modules. The source column identifies where each option is read.

For diagnostic options, `<diagnostic>` stands for the name of the configured diagnostic.

| Parameter Name | Description | Read in |
| --- | --- | --- |
| `<diagnostic>.center` | Centre coordinates of the plane sampled by a frame diagnostic. | `src/io/DiagFramePlane.cpp` |
| `<diagnostic>.deposit_fields` | Particle quantities to deposit onto the grid before producing a derived field. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.diagnostics` | Names of the configured diagnostics to run. | `src/simulation.hpp` |
| `<diagnostic>.field_name` | Field examined by a diagnostic filter. | `src/io/DiagFilter.cpp` |
| `<diagnostic>.field_names` | Fields to include in a frame or plotfile diagnostic. | `src/io/DiagFramePlane.cpp` |
| `<diagnostic>.file` | Output file name for the diagnostic. | `src/io/DiagBase.cpp` |
| `<diagnostic>.filters` | Filters applied before writing a diagnostic. | `src/io/DiagBase.cpp` |
| `<diagnostic>.include_fc_fields` | Include face-centred fields in a diagnostic plotfile. | `src/io/DiagPlotfile.cpp` |
| `<diagnostic>.int` | Diagnostic output interval in timesteps. | `src/io/DiagBase.cpp` |
| `<diagnostic>.log_spaced_bins` | Use logarithmically spaced PDF bins for an axis. | `src/io/DiagPDF.cpp` |
| `<diagnostic>.mass_max` | Upper particle-mass limit for particle deposition. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.mass_min` | Lower particle-mass limit for particle deposition. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.nBins` | Number of bins along a PDF axis. | `src/io/DiagPDF.cpp` |
| `<diagnostic>.nfiles` | Number of files used for a diagnostic plotfile. | `src/io/DiagPlotfile.cpp` |
| `<diagnostic>.normal` | Normal vector of a sampled frame plane. | `src/io/DiagFramePlane.cpp` |
| `<diagnostic>.normalization_expr` | Expression used to normalize deposited particle quantities. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.particle_types` | Particle species included in particle deposition. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.particles` | Particle species included in diagnostic plotfiles. | `src/io/DiagFramePlane.cpp` |
| `<diagnostic>.per` | Diagnostic output cadence selector. | `src/io/DiagBase.cpp` |
| `<diagnostic>.prefix` | Prefix for names of deposited particle fields. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.range` | Lower and upper bounds of a PDF axis. | `src/io/DiagPDF.cpp` |
| `<diagnostic>.t_age` | Maximum particle age included in deposition. | `src/io/DerivedParticleDeposition.cpp` |
| `<diagnostic>.time_int` | Diagnostic output interval in simulation time. | `src/io/DiagBase.cpp` |
| `<diagnostic>.type` | Selects the diagnostic implementation. | `src/simulation.hpp` |
| `<diagnostic>.value_greater` | Keep cells whose filtered field exceeds this value. | `src/io/DiagFilter.cpp` |
| `<diagnostic>.value_inrange` | Keep cells whose filtered field lies in this range. | `src/io/DiagFilter.cpp` |
| `<diagnostic>.value_less` | Keep cells whose filtered field is below this value. | `src/io/DiagFilter.cpp` |
| `<diagnostic>.var_names` | Names of the fields used as PDF axes. | `src/io/DiagPDF.cpp` |
| `<diagnostic>.weight_by` | PDF weighting: volume, mass, or cell count. | `src/io/DiagPDF.cpp` |
| `amr.max_level` | Maximum AMR refinement level; also sets a lower bound on grid-generation iterations. | `src/main.cpp` |
| `amr.n_cell` | Number of cells used by the BinaryOrbitCIC problem when constructing its grid. | `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp` |
| `atmosphere_scale_height` | Pressure scale height of the HydrostaticAtmosphere initial state. | `src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp` |
| `blast_problem.check_solution` | Enable the HydroBlast3D solution check. | `src/problems/HydroBlast3D/testHydroBlast3D.cpp` |
| `channel.rho0` | Initial mass density in the NSCBC channel test. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.s0` | Initial passive-scalar value in the NSCBC channel test. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.s_inflow` | Passive-scalar value prescribed at the NSCBC channel inflow. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.Tgas0` | Initial gas temperature in the NSCBC channel test. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.u0` | Initial longitudinal velocity in the NSCBC channel test. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.u_inflow` | Longitudinal velocity prescribed at the NSCBC channel inflow. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.v_inflow` | Transverse y velocity prescribed at the NSCBC channel inflow. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `channel.w_inflow` | Transverse z velocity prescribed at the NSCBC channel inflow. | `src/problems/NscbcChannel/testNscbcChannel.cpp` |
| `cloud_relpos_x` | Initial cloud x position as a fraction of the box length. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `conduction.conduction_cfl` | CFL factor used to limit the thermal-conduction timestep. | `src/QuokkaSimulation.hpp` |
| `conduction.conductivity_prefactor` | Prefactor multiplying the thermal conductivity. | `src/QuokkaSimulation.hpp` |
| `conduction.enabled` | Enable thermal conduction. | `src/QuokkaSimulation.hpp` |
| `conduction.flux_limiter_phi` | Flux-limiter coefficient for thermal conduction. | `src/QuokkaSimulation.hpp` |
| `conduction.saturation_factor` | Factor controlling saturation of the conductive heat flux. | `src/QuokkaSimulation.hpp` |
| `cooling.cooling_table_type` | Cooling-table format; only `resampled` is supported. | `src/QuokkaSimulation.hpp` |
| `cooling_test.min_spatial_heating_T_ratio` | Minimum spatial-heating temperature ratio used by the resampled-cooling test. | `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp` |
| `cooling_test.output_csv_file` | CSV file written by the resampled-cooling test. | `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp` |
| `cooling_test.reference_solution_file` | Reference solution file read by the resampled-cooling test. | `src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp` |
| `density_refinement` | Enable density-based refinement around particle sinks. | `src/problems/ParticleSinkSubcycle/testParticleSinkSubcycle.cpp` |
| `disk_galaxy.disk_gas_mass_Msun` | Total initial gas mass of the disk, in solar masses. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.disk_perturb_amplitude` | Amplitude of the initial disk perturbation. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.disk_perturb_Rmax_kpc` | Maximum radius of the initial disk perturbation, in kpc. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.disk_Rscale_kpc` | Radial exponential scale length of the gas disk, in kpc. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.disk_temperature` | Initial gas temperature of the disk. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.disk_zscale_kpc` | Vertical scale height of the gas disk, in kpc. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.flux_sphere_radius_kpc` | Radius of the sphere used to measure fluxes, in kpc. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.halo_vphi_expr` | Expression for the halo azimuthal velocity. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.initial_scalar_density` | Initial density assigned to the passive scalar. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.magnetic_field_microgauss` | Initial magnetic-field strength, in microgauss. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.particle_file` | Input file containing the initial particle population. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.refine_Rmax_kpc` | Maximum cylindrical radius of the refined disk region, in kpc. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.refine_zmax_kpc` | Maximum height of the refined disk region, in kpc. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `disk_galaxy.vcirc_file` | Input file containing the circular-velocity profile. | `src/problems/DiskGalaxy/testDiskGalaxy.cpp` |
| `do_frame_shift` | Shift the shock-cloud simulation frame with the flow. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `geometry.prob_hi` | Upper physical-domain coordinates used by the MHD dynamo test. | `src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp` |
| `geometry.prob_lo` | Lower physical-domain coordinates used by the MHD dynamo test. | `src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp` |
| `hydro.bulk_viscosity` | Constant bulk-viscosity coefficient when the constant-viscosity model is selected. | `src/QuokkaSimulation.hpp` |
| `hydro.shear_viscosity` | Constant shear-viscosity coefficient when the constant-viscosity model is selected. | `src/QuokkaSimulation.hpp` |
| `ignore_return` | Return success from the executable even if the problem test returns a failure code. | `src/main.cpp` |
| `jeansRefine.density_threshold` | Density threshold for Jeans-based refinement in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `jeansRefine.ncells` | Target number of cells resolving the Jeans length. | `src/problems/PopIII/testPopIII.cpp` |
| `kappa0` | Opacity coefficient used by the radiation-hydrodynamic pulse tests. | `src/problems/RadhydroPulse/testRadhydroPulse.cpp` |
| `list_options` | Print the compiled Quokka option registry and exit. | `src/main.cpp` |
| `Mach_shock` | Mach number of the incident shock in the ShockCloud problem. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `marshak.use_wavespeed_correction` | Enable the wavespeed correction in the asymptotic Marshak-wave test. | `src/problems/RadMarshakAsymptotic/testRadMarshakAsymptotic.cpp` |
| `max_t_cc` | Maximum simulation time expressed in cloud-crushing times. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `max_time` | Maximum simulation time used by the radiation-streaming tests. | `src/problems/RadStreamingY/testRadStreamingY.cpp` |
| `mhd.emf_compute_scheme` | Algorithm used to compute edge-centred electromotive forces. | `src/QuokkaSimulation.hpp` |
| `mhd.resistivity` | Constant magnetic resistivity when the constant-resistivity model is selected. | `src/QuokkaSimulation.hpp` |
| `network.energy_switch` | Enable energy coupling in the one-zone photoionization network. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `network.recombination_switch` | Enable recombinations in the Strömgren-sphere network. | `src/problems/StromgrenSphere/testStromgrenSphere.cpp` |
| `nH_bg` | Background hydrogen number density in the ShockCloud problem. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `nH_cloud` | Cloud hydrogen number density in the ShockCloud problem. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `P_over_k` | Gas pressure divided by Boltzmann's constant in the ShockCloud problem. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `particles.low_mass_composite_max_mass` | Maximum mass represented by a low-mass composite stellar particle. | `src/particles/particle_types.hpp` |
| `particles.SN_smooth_gas_velocity` | Smooth gas velocity in the supernova deposition stencil to enforce energy conservation. | `src/particles/particle_types.hpp` |
| `particles.scalar_yield_per_SN` | Passive-scalar yield assigned per supernova. | `src/particles/particle_types.hpp` |
| `perturb.cloud_density` | Density of the perturbed cloud in the StarCluster problem. | `src/problems/StarCluster/testStarCluster.cpp` |
| `perturb.cloud_numdens` | Number density of the perturbed cloud in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `perturb.cloud_omega` | Angular velocity imposed on the perturbed cloud. | `src/problems/PopIII/testPopIII.cpp` |
| `perturb.cloud_radius` | Radius of the perturbed cloud. | `src/problems/PopIII/testPopIII.cpp` |
| `perturb.filename` | Input file used to seed the PopIII perturbation. | `src/problems/PopIII/testPopIII.cpp` |
| `perturb.rms_velocity` | RMS velocity of the seeded perturbation. | `src/problems/PopIII/testPopIII.cpp` |
| `perturb.virial_parameter` | Initial virial parameter of the StarCluster cloud. | `src/problems/StarCluster/testStarCluster.cpp` |
| `photochemistry.enabled` | Enable photochemistry source terms. | `src/QuokkaSimulation.hpp` |
| `photochemistry.max_density_allowed` | Upper gas-density limit for photochemistry updates. | `src/QuokkaSimulation.hpp` |
| `photochemistry.min_density_allowed` | Lower gas-density limit for photochemistry updates. | `src/QuokkaSimulation.hpp` |
| `photoionization.n_e_init` | Sets the initial electron number density in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.n_HI_init` | Sets the initial neutral-hydrogen number density in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.n_HII_init` | Sets the initial ionized-hydrogen number density in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.n_photon` | Sets the initial photon number density in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.small_dens` | Sets the minimum density used by the initial state in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.small_temp` | Sets the minimum temperature used by the initial state in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.temperature` | Sets the initial gas temperature in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization.tend` | Sets the end time of the test in the one-zone photoionization test. | `src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp` |
| `photoionization_momentum.n_e_init` | Sets the initial electron number density in the photoionization momentum test. | `src/problems/DTypeFrontVC/testDTypeFrontVC.cpp` |
| `photoionization_momentum.n_HI_init` | Sets the initial neutral-hydrogen number density in the photoionization momentum test. | `src/problems/DTypeFrontVC/testDTypeFrontVC.cpp` |
| `photoionization_momentum.n_HII_init` | Sets the initial ionized-hydrogen number density in the photoionization momentum test. | `src/problems/DTypeFrontVC/testDTypeFrontVC.cpp` |
| `photoionization_momentum.small_dens` | Sets the minimum density used by the initial state in the photoionization momentum test. | `src/problems/DTypeFrontVC/testDTypeFrontVC.cpp` |
| `photoionization_momentum.small_temp` | Sets the minimum temperature used by the initial state in the photoionization momentum test. | `src/problems/DTypeFrontVC/testDTypeFrontVC.cpp` |
| `photoionization_momentum.temperature` | Sets the initial gas temperature in the photoionization momentum test. | `src/problems/DTypeFrontVC/testDTypeFrontVC.cpp` |
| `photoionize.beamed` | Inject each wing of the radiation source outward when enabled; inject isotropically when disabled. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.flux` | Incident radiation flux in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.flux_ion` | Incident ionizing-photon flux in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.kappa1` | First opacity coefficient used by the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.kappa2` | Second opacity coefficient used by the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.n_e_init` | Sets the initial electron number density in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.n_HI_init` | Sets the initial neutral-hydrogen number density in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.n_HII_init` | Sets the initial ionized-hydrogen number density in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.small_dens` | Sets the minimum density used by the initial state in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.small_temp` | Sets the minimum temperature used by the initial state in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.source_cells` | Number of source cells on each side of the central radiation slab. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.T_dust_destroy` | Dust-destruction temperature in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `photoionize.temperature` | Sets the initial gas temperature in the DTypeFront1D test. | `src/problems/DTypeFront1D/testDTypeFront1D.cpp` |
| `primordial_chem.primary_species_1` | Initial abundance of primordial-chemistry primary species 1 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_10` | Initial abundance of primordial-chemistry primary species 10 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_11` | Initial abundance of primordial-chemistry primary species 11 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_12` | Initial abundance of primordial-chemistry primary species 12 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_13` | Initial abundance of primordial-chemistry primary species 13 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_14` | Initial abundance of primordial-chemistry primary species 14 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_2` | Initial abundance of primordial-chemistry primary species 2 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_3` | Initial abundance of primordial-chemistry primary species 3 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_4` | Initial abundance of primordial-chemistry primary species 4 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_5` | Initial abundance of primordial-chemistry primary species 5 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_6` | Initial abundance of primordial-chemistry primary species 6 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_7` | Initial abundance of primordial-chemistry primary species 7 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_8` | Initial abundance of primordial-chemistry primary species 8 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.primary_species_9` | Initial abundance of primordial-chemistry primary species 9 in the PopIII problem. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.small_dens` | Density floor used by the PopIII primordial-chemistry setup. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.small_temp` | Temperature floor used by the PopIII primordial-chemistry setup. | `src/problems/PopIII/testPopIII.cpp` |
| `primordial_chem.temperature` | Initial gas temperature in the PopIII chemistry setup. | `src/problems/PopIII/testPopIII.cpp` |
| `problem.boost_vel_x` | x velocity added to the ParticleSink test setup. | `src/problems/ParticleSink/testParticleSink.cpp` |
| `problem.boost_velocity` | Three-component velocity boost applied to the RandomBlast setup. | `src/problems/RandomBlast/testRandomBlast.cpp` |
| `problem.do_split_particles` | Enable particle splitting in the BinaryOrbitCIC test. | `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp` |
| `problem.flux_source` | Use a flux-based radiation source in the RadStreaming test. | `src/problems/RadStreaming/testRadStreaming.cpp` |
| `problem.history_dt_over_ts0` | Sampling interval for RDI history output, in initial stopping times. | `src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp` |
| `problem.hot_T` | Temperature of the hot phase in the TallBoxSf setup. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.IC_file` | Input file containing TallBoxSf initial conditions. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.initial_scalar_density` | Initial passive-scalar density in the TallBoxSf setup. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.kappa1` | First dust opacity coefficient in the Marshak-wave test. | `src/problems/RadMarshakDust/testRadMarshakDust.cpp` |
| `problem.kappa2` | Second dust opacity coefficient in the Marshak-wave test. | `src/problems/RadMarshakDust/testRadMarshakDust.cpp` |
| `problem.M0_in_Msun` | Initial stellar mass for the particle-evolution test, in solar masses. | `src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp` |
| `problem.n0` | Initial gas number density in the ParticleSF test. | `src/problems/ParticleSF/testParticleSF.cpp` |
| `problem.n_amb` | Ambient gas number density in the RandomBlast test. | `src/problems/RandomBlast/testRandomBlast.cpp` |
| `problem.noise_amplitude` | Amplitude of the initial RDI noise perturbation. | `src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp` |
| `problem.noise_seed` | Random seed for the initial RDI noise perturbation. | `src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp` |
| `problem.num_particles` | Number of particles created by the SphericalCollapse test. | `src/problems/SphericalCollapse/testSphericalCollapse.cpp` |
| `problem.part_fn` | ASCII file used to initialize the RandomBlast stellar particles. | `src/problems/RandomBlast/testRandomBlast.cpp` |
| `problem.particles_file` | Input particle file for the ParticleSink test. | `src/problems/ParticleSink/testParticleSink.cpp` |
| `problem.particles_filename` | Input particle file for the ParticleRadiation test. | `src/problems/ParticleRadiation/testParticleRadiation.cpp` |
| `problem.reference_steps` | Reference number of steps used by the DustyAlfvenWave test. | `src/problems/DustyAlfvenWave/testDustyAlfvenWave.cpp` |
| `problem.refine_center` | Centre of the refined region in the ParticleAccretion test. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.refine_half_domain` | Refine half of the domain in the ParticleCreation test. | `src/problems/ParticleCreation/testParticleCreation.cpp` |
| `problem.refine_zmax` | Maximum height of the refined region in TallBoxSf. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.return_1_at_fail` | Return a failure status when the ParticleAccretion check fails. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.rho0` | Initial gas density in the ParticleAccretion test. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.rho01` | Initial density of the first gas phase in TallBoxSf. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.seed` | Random seed for particle placement in SphericalCollapse. | `src/problems/SphericalCollapse/testSphericalCollapse.cpp` |
| `problem.sigma1` | Velocity dispersion of the first gas phase in TallBoxSf. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.sink_file` | Input sink-particle file for the ParticleAccretion test. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.SN_particles_file` | Input supernova-particle file for the SN test. | `src/problems/SN/testSN.cpp` |
| `problem.split_factor` | Number of child particles created per split in BinaryOrbitCIC. | `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp` |
| `problem.stage_times_over_ts0` | RDI stage times expressed in initial stopping times. | `src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp` |
| `problem.star_mass` | Stellar mass used by the ParticleAccretion test. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.stars_file` | Input stellar-particle file for TallBoxSf. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.T_amb` | Ambient gas temperature in the RandomBlast test. | `src/problems/RandomBlast/testRandomBlast.cpp` |
| `problem.t_end_over_t_b` | End time expressed in the ParticleAccretion reference timescale. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.Tamb` | Ambient gas temperature in the ParticleSF test. | `src/problems/ParticleSF/testParticleSF.cpp` |
| `problem.turnon_fextract` | Enable the fextract diagnostic in ParticleAccretion. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.uniform_density` | Use a uniform initial gas density in ParticleAccretion. | `src/problems/ParticleAccretion/testParticleAccretion.cpp` |
| `problem.validate_initial_imf_stats` | Check the initial stellar initial-mass-function statistics in ParticleSF. | `src/problems/ParticleSF/testParticleSF.cpp` |
| `problem.verify_low_mass_cap_on_restart` | Check the low-mass composite-particle cap after restart in ParticleSF. | `src/problems/ParticleSF/testParticleSF.cpp` |
| `problem.verify_particle_layout` | Check the particle layout in the BinaryOrbitCIC test. | `src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp` |
| `problem.warm_T` | Temperature of the warm phase in TallBoxSf. | `src/problems/TallBoxSf/testTallBoxSf.cpp` |
| `problem.write_csv` | Write CSV output from the problem test. | `src/problems/DustDampedGyromotion/testDustDampedGyromotion.cpp` |
| `quokka_time_units.Gyr` | Number of seconds in one gigayear for parser time expressions. | `src/simulation.hpp` |
| `quokka_time_units.kyr` | Number of seconds in one kiloyear for parser time expressions. | `src/simulation.hpp` |
| `quokka_time_units.Myr` | Number of seconds in one megayear for parser time expressions. | `src/simulation.hpp` |
| `quokka_time_units.yr` | Number of seconds in one year for parser time expressions. | `src/simulation.hpp` |
| `R_cloud_pc` | Cloud radius, in parsecs, in the ShockCloud problem. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `setup.advection` | Enable vortex advection in the MHDBalsaraVortex test. | `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp` |
| `setup.advection_angle_deg` | Angle of the FieldLoop advection velocity, in degrees. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.advection_vz` | z component of the FieldLoop advection velocity. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.angle_between_k_b0` | Angle between the wavevector and background magnetic field, in degrees. | `src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp` |
| `setup.error_tol` | Error tolerance used by the HydroShearWave test. | `src/problems/HydroShearWave/testHydroShearWave.cpp` |
| `setup.loop_center_x` | x coordinate of the FieldLoop centre. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.loop_center_y` | y coordinate of the FieldLoop centre. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.loop_radius` | Radius of the magnetic field loop. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.machine_precision_target` | Target accuracy for the HydroWaveConvergence test. | `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` |
| `setup.num_modes_x` | Number of Alfvén-wave modes along x. | `src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp` |
| `setup.num_modes_y` | Number of Alfvén-wave modes along y. | `src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp` |
| `setup.num_modes_z` | Number of Alfvén-wave modes along z. | `src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp` |
| `setup.num_periods` | Number of FieldLoop advection periods to run. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.nx_max` | Largest x resolution in the HydroWaveConvergence sweep. | `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` |
| `setup.nx_start` | Starting x resolution in the HydroWaveConvergence sweep. | `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` |
| `setup.refine_based_on` | Field or criterion used to trigger refinement in FieldLoop. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.refine_n_dims` | Number of dimensions refined in the HydroWaveConvergence test. | `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` |
| `setup.region_hi_x` | Upper x boundary of the FieldLoop refinement region. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.region_hi_y` | Upper y boundary of the FieldLoop refinement region. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.region_hi_z` | Upper z boundary of the FieldLoop refinement region. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.region_lo_x` | Lower x boundary of the FieldLoop refinement region. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.region_lo_y` | Lower y boundary of the FieldLoop refinement region. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.region_lo_z` | Lower z boundary of the FieldLoop refinement region. | `src/problems/FieldLoop/testFieldLoop.cpp` |
| `setup.run_convergence` | Run the HydroWaveConvergence resolution study. | `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` |
| `setup.run_sim` | Run the HydroWaveConvergence simulation. | `src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp` |
| `setup.seed_b_fraction` | Fractional amplitude of the seed magnetic field in the dynamo test. | `src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp` |
| `setup.seed_b_wavenumber` | Wavenumber of the seed magnetic field in the dynamo test. | `src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp` |
| `setup.shear_flow_axis` | Axis of the imposed shear flow in HydroShearWave. | `src/problems/HydroShearWave/testHydroShearWave.cpp` |
| `setup.shear_grad_axis` | Axis along which the HydroShearWave flow varies. | `src/problems/HydroShearWave/testHydroShearWave.cpp` |
| `setup.vortex_b_magn` | Magnetic-field strength of the MHDBalsaraVortex initial state. | `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp` |
| `setup.vortex_Mach` | Mach number of the MHDBalsaraVortex initial state. | `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp` |
| `setup.vortex_radius` | Radius of the MHDBalsaraVortex initial state. | `src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp` |
| `sharp_cloud_edge` | Use a sharp rather than smoothed cloud boundary. | `src/problems/ShockCloud/testShockCloud.cpp` |
| `shell_problem.filename` | Input filename used by the radiation-hydrodynamic shell problem. | `src/problems/RadhydroShell/testRadhydroShell.cpp` |
| `stromgen.n_e_init` | Sets the initial electron number density in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.n_HI_init` | Sets the initial neutral-hydrogen number density in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.n_HII_init` | Sets the initial ionized-hydrogen number density in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.Q` | Sets the ionizing-photon emission rate in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.small_dens` | Sets the minimum density used by the initial state in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.small_temp` | Sets the minimum temperature used by the initial state in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.temperature` | Sets the initial gas temperature in the Strömgren-front test. | `src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp` |
| `stromgen.tend` | Sets the end time of the test in the Strömgren-front test. | `src/problems/StromgrenSphere/testStromgrenSphere.cpp` |
| `v0_adv` | Advection velocity of the radiation-hydrodynamic pulse. | `src/problems/RadhydroPulse/testRadhydroPulse.cpp` |
| `vortex.P0` | Initial gas pressure in the NSCBC vortex test. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
| `vortex.s_inflow` | Passive-scalar value prescribed at the NSCBC vortex inflow. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
| `vortex.strength` | Amplitude of the imposed NSCBC vortex. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
| `vortex.Tgas0` | Initial gas temperature in the NSCBC vortex test. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
| `vortex.u_inflow` | Longitudinal inflow velocity in the NSCBC vortex test. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
| `vortex.v_inflow` | Transverse y inflow velocity in the NSCBC vortex test. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
| `vortex.w_inflow` | Transverse z inflow velocity in the NSCBC vortex test. | `src/problems/NscbcVortex/testNscbcVortex.cpp` |
