#ifndef QUOKKA_PARM_PARSE_OPTION_REGISTRY_HPP_
#define QUOKKA_PARM_PARSE_OPTION_REGISTRY_HPP_

#include <cstddef>
#include <ostream>
#include <string>
#include <string_view>

// Full names are checked in. Add new options here with a useful description.
// The asterisk prefix denotes a runtime diagnostic prefix.
#define QUOKKA_PARM_PARSE_OPTIONS(X)                                                                                                                           \
	X("", "Mach_shock", "Mach number of the incident shock in the ShockCloud problem.", "src/problems/ShockCloud/testShockCloud.cpp")                      \
	X("", "P_over_k", "Gas pressure divided by Boltzmann's constant in the ShockCloud problem.", "src/problems/ShockCloud/testShockCloud.cpp")             \
	X("", "R_cloud_pc", "Cloud radius, in parsecs, in the ShockCloud problem.", "src/problems/ShockCloud/testShockCloud.cpp")                              \
	X("", "amr_interpolation_method",                                                                                                                      \
	  "Selects the method used to interpolate from coarse to fine AMR levels. 0: piecewise constant, 1: linear. Except for debugging, this should not be " \
	  "changed.",                                                                                                                                          \
	  "src/simulation.hpp")                                                                                                                                \
	X("", "atmosphere_scale_height", "Pressure scale height of the HydrostaticAtmosphere initial state.",                                                  \
	  "src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp")                                                                                  \
	X("", "cfl", "Sets the CFL number for the simulation.", "src/simulation.hpp")                                                                          \
	X("", "checkpoint_interval", "The number of coarse timesteps between checkpoint outputs.", "src/simulation.hpp")                                       \
	X("", "checkpoint_prefix", "The prefix for checkpoint output filenames.", "src/simulation.hpp")                                                        \
	X("", "cloud_relpos_x", "Initial cloud x position as a fraction of the box length.", "src/problems/ShockCloud/testShockCloud.cpp")                     \
	X("", "const_sfr_Msun_per_year_per_kpc2", "Constant star formation rate in Msun/year/kpc^2. If non-negative, overrides the simulation SFR.",           \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("", "debug_density_floor_plot",                                                                                                                      \
	  "If set to 1, adds a derived field `density_floor_dbg` to plotfiles to visualize the spatially varying density floor.", "src/simulation.hpp")        \
	X("", "density_floor", "The minimum density value allowed in the simulation. Enforced through EnforceLimits.",                                         \
	  "src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp;src/simulation.hpp")                                                               \
	X("", "density_floor_expr",                                                                                                                            \
	  "Optional AMReX parser expression for a spatially varying density floor. Variables: x, y, z, base_density_floor. When set, this overrides the "      \
	  "constant floor.",                                                                                                                                   \
	  "src/problems/HydrostaticAtmosphere/testHydrostaticAtmosphere.cpp;src/simulation.hpp")                                                               \
	X("", "density_refinement", "Enable density-based refinement around particle sinks.",                                                                  \
	  "src/problems/ParticleSinkSubcycle/testParticleSinkSubcycle.cpp")                                                                                    \
	X("", "do_frame_shift", "Shift the shock-cloud simulation frame with the flow.", "src/problems/ShockCloud/testShockCloud.cpp")                         \
	X("", "do_reflux",                                                                                                                                     \
	  "This turns on refluxing at coarse-fine boundaries (1) or turns it off (0). Except for debugging, this should always be on when AMR is used.",       \
	  "src/simulation.hpp")                                                                                                                                \
	X("", "do_subcycle", "This turns on subcycling at coarse-fine boundaries (1) or turns it off (0).", "src/simulation.hpp")                              \
	X("", "do_tracers", "This turns on tracer particles. They are initialized one-per-cell and they follow the fluid velocity.", "src/simulation.hpp")     \
	X("", "heating_rate_external",                                                                                                                         \
	  "Optional AMReX parser expression for external heating rate per H atom (erg/s/H). Variables: `x`, `y`, `z`, `time`, `dt`. Evaluated at cell "        \
	  "centres, so the rate may vary in both space and time. Effective when cooling is enabled.",                                                          \
	  "src/simulation.hpp")                                                                                                                                \
	X("", "ignore_return", "Return success from the executable even if the problem test returns a failure code.", "src/main.cpp")                          \
	X("", "init_shrink", "Factor to shrink the initial timestep by.", "src/simulation.hpp")                                                                \
	X("", "kappa0", "Opacity coefficient used by the radiation-hydrodynamic pulse tests.",                                                                 \
	  "src/problems/RadhydroPulse/testRadhydroPulse.cpp;src/problems/RadhydroPulseDyn/testRadhydroPulseDyn.cpp;src/problems/RadhydroPulseGrey/"            \
	  "testRadhydroPulseGrey.cpp")                                                                                                                         \
	X("", "max_dt", "Optional maximum timestep.", "src/problems/RadForce/testRadForce.cpp;src/simulation.hpp")                                             \
	X("", "max_t_cc", "Maximum simulation time expressed in cloud-crushing times.", "src/problems/ShockCloud/testShockCloud.cpp")                          \
	X("", "max_time", "Maximum simulation time used by the radiation-streaming tests.",                                                                    \
	  "src/problems/RadStreamingY/testRadStreamingY.cpp;src/problems/RadhydroPulse/testRadhydroPulse.cpp;src/problems/RadhydroPulseDyn/"                   \
	  "testRadhydroPulseDyn.cpp;src/problems/RadhydroPulseGrey/testRadhydroPulseGrey.cpp")                                                                 \
	X("", "max_timesteps", "The maximum number of time steps for the simulation.", "src/simulation.hpp")                                                   \
	X("", "max_walltime",                                                                                                                                  \
	  "The maximum walltime for the simulation in the format DD:HH:SS (days/hours/seconds). After 90% of this walltime elapses, the simulation will "      \
	  "automatically stop and exit.",                                                                                                                      \
	  "src/simulation.hpp")                                                                                                                                \
	X("", "nH_bg", "Background hydrogen number density in the ShockCloud problem.", "src/problems/ShockCloud/testShockCloud.cpp")                          \
	X("", "nH_cloud", "Cloud hydrogen number density in the ShockCloud problem.", "src/problems/ShockCloud/testShockCloud.cpp")                            \
	X("", "particle_cfl", "Sets the CFL number for particle advection. This is independent of the hydro CFL number.", "src/simulation.hpp")                \
	X("", "particle_speed_abort", "If value > 0 and the max particle speed exceeds this value, the simulation aborts.", "src/simulation.hpp")              \
	X("", "plotfile_interval", "The number of coarse timesteps between plotfile outputs.", "src/simulation.hpp")                                           \
	X("", "plotfile_prefix", "The prefix for plotfile output filenames.", "src/simulation.hpp")                                                            \
	X("", "poisson_abstol", "Absolute tolerance for the Poisson solver convergence (scaled by minimum RHS value).", "src/simulation.hpp")                  \
	X("", "poisson_reltol", "Relative tolerance for the Poisson solver convergence.", "src/simulation.hpp")                                                \
	X("", "poisson_supercycle_interval", "The number of coarse timesteps between Poisson supercycle operations.", "src/simulation.hpp")                    \
	X("", "print_cycle_timing", "If set to 1, prints per-cycle timing information.", "src/simulation.hpp")                                                 \
	X("", "regrid_interval", "The number of timesteps between AMR regridding.", "src/simulation.hpp")                                                      \
	X("", "restartfile", "The path to a checkpoint file from which to restart the simulation.",                                                            \
	  "src/problems/ParticleSF/testParticleSF.cpp;src/simulation.hpp")                                                                                     \
	X("", "sf_area_kpc2", "Area of the star formation region in kpc^2.", "src/QuokkaSimulation.hpp")                                                       \
	X("", "sfh_interval", "Interval for updating/writing star formation history.", "src/simulation.hpp")                                                   \
	X("", "sfh_to_pe_heating_table", "Path to the table converting star formation history to photoelectric heating.", "src/QuokkaSimulation.hpp")          \
	X("", "sharp_cloud_edge", "Use a sharp rather than smoothed cloud boundary.", "src/problems/ShockCloud/testShockCloud.cpp")                            \
	X("", "show_performance_hints", "If set to 1, prints performance hints.", "src/simulation.hpp")                                                        \
	X("", "signal_speed_abort", "If value > 0 and the max signal speed exceeds this value, the simulation aborts.", "src/simulation.hpp")                  \
	X("", "skip_initial_plotfile", "Skip writing the initial plotfile at t=0.", "src/simulation.hpp")                                                      \
	X("", "statistics_file", "The prefix for the statistics output file.", "src/simulation.hpp")                                                           \
	X("", "statistics_interval", "The number of coarse timesteps between statistics outputs.", "src/simulation.hpp")                                       \
	X("", "stop_time", "The simulation time at which to stop evolving the simulation.",                                                                    \
	  "src/problems/FieldLoop/testFieldLoop.cpp;src/problems/MHDAlfvenWaveCircularConvergence/testMHDAlfvenWaveCircularConvergence.cpp;src/problems/"      \
	  "MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp;src/problems/"         \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp;src/simulation.hpp")                                                                          \
	X("", "suppress_output", "If set to 1, this disables output to stdout while the simulation is running.", "src/simulation.hpp")                         \
	X("", "temperature_floor",                                                                                                                             \
	  "The minimum temperature allowed in the simulation, in kelvin for the `CGS` unit system and in code units for the `CONSTANTS` and `CUSTOM` unit "    \
	  "systems. Enforced through EnforceLimits. The default is 2.7 K (the CMB temperature) when `Physics_Traits<problem_t>::unit_system` is "              \
	  "`UnitSystem::CGS`, where the value is unambiguously in kelvin. It defaults to 0 for `CONSTANTS` and `CUSTOM`, because there a literal value is in " \
	  "code units rather than kelvin: `CONSTANTS` fixes the physical constants without defining a temperature scale, and `CUSTOM` rescales temperature "   \
	  "by `Physics_Traits::unit_temperature`. Problems in those unit systems should set this explicitly if they need a floor. Idealized, dimensionless "   \
	  "test problems should declare `UnitSystem::CONSTANTS` rather than relying on a CGS default.",                                                        \
	  "src/simulation.hpp")                                                                                                                                \
	X("", "use_sfh_based_pe_heating", "If set to 1, enables photoelectric heating based on star formation history.",                                       \
	  "src/QuokkaSimulation.hpp;src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp")                                                           \
	X("", "v0_adv", "Advection velocity of the radiation-hydrodynamic pulse.",                                                                             \
	  "src/problems/RadhydroPulse/testRadhydroPulse.cpp;src/problems/RadhydroPulseDyn/testRadhydroPulseDyn.cpp;src/problems/RadhydroPulseGrey/"            \
	  "testRadhydroPulseGrey.cpp")                                                                                                                         \
	X("*", "field_name", "Field examined by a diagnostic filter.", "src/io/DiagFilter.cpp")                                                                \
	X("*", "file", "Output file name for the diagnostic.", "src/io/DiagBase.cpp")                                                                          \
	X("*", "include_fc_fields", "Include face-centred fields in a diagnostic plotfile.", "src/io/DiagPlotfile.cpp")                                        \
	X("*", "int", "Diagnostic output interval in timesteps.", "src/io/DiagBase.cpp")                                                                       \
	X("*", "log_spaced_bins", "Use logarithmically spaced PDF bins for an axis.", "src/io/DiagPDF.cpp")                                                    \
	X("*", "mass_max", "Upper particle-mass limit for particle deposition.", "src/io/DerivedParticleDeposition.cpp")                                       \
	X("*", "mass_min", "Lower particle-mass limit for particle deposition.", "src/io/DerivedParticleDeposition.cpp")                                       \
	X("*", "nfiles", "Number of files used for a diagnostic plotfile.", "src/io/DiagPlotfile.cpp")                                                         \
	X("*", "normalization_expr", "Expression used to normalize deposited particle quantities.", "src/io/DerivedParticleDeposition.cpp")                    \
	X("*", "per", "Diagnostic output cadence selector.", "src/io/DiagBase.cpp")                                                                            \
	X("*", "prefix", "Prefix for names of deposited particle fields.", "src/io/DerivedParticleDeposition.cpp")                                             \
	X("*", "t_age", "Maximum particle age included in deposition.", "src/io/DerivedParticleDeposition.cpp")                                                \
	X("*", "weight_by", "PDF weighting: volume, mass, or cell count.", "src/io/DiagPDF.cpp")                                                               \
	X("amr", "checkpoint_nfiles", "Maximum number of binary files per multifab for checkpoints. Controls parallel I/O chunking.", "src/simulation.hpp")    \
	X("amr", "max_level", "Maximum AMR refinement level; also sets a lower bound on grid-generation iterations.", "src/main.cpp")                          \
	X("amr", "n_cell", "Number of cells used by the BinaryOrbitCIC problem when constructing its grid.",                                                   \
	  "src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp;src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp;src/problems/StromgrenSphere/"  \
	  "testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/testStromgrenSphereRSLA.cpp")                                                              \
	X("amr", "plot_nfiles", "Maximum number of binary files per multifab for plotfiles. Controls parallel I/O chunking.", "src/simulation.hpp")            \
	X("blast_problem", "check_solution", "Enable the HydroBlast3D solution check.", "src/problems/HydroBlast3D/testHydroBlast3D.cpp")                      \
	X("channel", "Tgas0", "Initial gas temperature in the NSCBC channel test.", "src/problems/NscbcChannel/testNscbcChannel.cpp")                          \
	X("channel", "rho0", "Initial mass density in the NSCBC channel test.", "src/problems/NscbcChannel/testNscbcChannel.cpp")                              \
	X("channel", "s0", "Initial passive-scalar value in the NSCBC channel test.", "src/problems/NscbcChannel/testNscbcChannel.cpp")                        \
	X("channel", "s_inflow", "Passive-scalar value prescribed at the NSCBC channel inflow.", "src/problems/NscbcChannel/testNscbcChannel.cpp")             \
	X("channel", "u0", "Initial longitudinal velocity in the NSCBC channel test.", "src/problems/NscbcChannel/testNscbcChannel.cpp")                       \
	X("channel", "u_inflow", "Longitudinal velocity prescribed at the NSCBC channel inflow.", "src/problems/NscbcChannel/testNscbcChannel.cpp")            \
	X("channel", "v_inflow", "Transverse y velocity prescribed at the NSCBC channel inflow.", "src/problems/NscbcChannel/testNscbcChannel.cpp")            \
	X("channel", "w_inflow", "Transverse z velocity prescribed at the NSCBC channel inflow.", "src/problems/NscbcChannel/testNscbcChannel.cpp")            \
	X("chemistry", "enabled", "If set to 1, turns on chemistry as a Strang-split source term.", "src/QuokkaSimulation.hpp")                                \
	X("chemistry", "max_density_allowed",                                                                                                                  \
	  "Maximum density value for which chemistry calculations are accurate. Chemistry is not performed for cells with densities above this threshold.",    \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("chemistry", "min_density_allowed",                                                                                                                  \
	  "Minimum density value for which chemistry calculations are performed. Chemistry is not performed for cells with densities below this threshold.",   \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("conduction", "conduction_cfl", "CFL factor used to limit the thermal-conduction timestep.", "src/QuokkaSimulation.hpp")                             \
	X("conduction", "conductivity_prefactor", "Prefactor multiplying the thermal conductivity.", "src/QuokkaSimulation.hpp")                               \
	X("conduction", "enabled", "Enable thermal conduction.", "src/QuokkaSimulation.hpp")                                                                   \
	X("conduction", "flux_limiter_phi", "Flux-limiter coefficient for thermal conduction.", "src/QuokkaSimulation.hpp")                                    \
	X("conduction", "saturation_factor", "Factor controlling saturation of the conductive heat flux.", "src/QuokkaSimulation.hpp")                         \
	X("cooling", "cooling_table_type", "Table type. Only `\"resampled\"` is supported.", "src/QuokkaSimulation.hpp;src/problems/SN/testSN.cpp")            \
	X("cooling", "enabled",                                                                                                                                \
	  "Only takes effect when the problem sets `EOSBackend = EOSTabulated<P>` (i.e. `quokka::EOS<P>::is_tabulated`). If set to 0, disables the cooling "   \
	  "integrator (`applyCooling`) while the tabulated EOS still uses the table to compute temperature — useful for testing. Has no effect otherwise: "    \
	  "for non-tabulated EOS backends, the cooling integrator never runs regardless of this value.",                                                       \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("cooling", "hdf5_data_file",                                                                                                                         \
	  "The path to the cooling tables in HDF5 format. We recommend using `extern/cooling/CloudyData_UVB=HM2012_resampled.h5` for ISM at solar "            \
	  "metallicity.",                                                                                                                                      \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("cooling", "read_tables_even_if_disabled",                                                                                                           \
	  "If set to 1, reads the cooling tables even if the problem does not use the `EOSTabulated` backend. Not needed for problems that set `EOSBackend = " \
	  "EOSTabulated<P>`.",                                                                                                                                 \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("cooling_test", "min_spatial_heating_T_ratio", "Minimum spatial-heating temperature ratio used by the resampled-cooling test.",                      \
	  "src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp")                                                                                    \
	X("cooling_test", "output_csv_file", "CSV file written by the resampled-cooling test.",                                                                \
	  "src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp")                                                                                    \
	X("cooling_test", "reference_solution_file", "Reference solution file read by the resampled-cooling test.",                                            \
	  "src/problems/ResampledCoolingTest/testResampledCoolingTest.cpp")                                                                                    \
	X("disk_galaxy", "disk_Rscale_kpc", "Radial exponential scale length of the gas disk, in kpc.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")          \
	X("disk_galaxy", "disk_gas_mass_Msun", "Total initial gas mass of the disk, in solar masses.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")           \
	X("disk_galaxy", "disk_perturb_Rmax_kpc", "Maximum radius of the initial disk perturbation, in kpc.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")    \
	X("disk_galaxy", "disk_perturb_amplitude", "Amplitude of the initial disk perturbation.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")                \
	X("disk_galaxy", "disk_temperature", "Initial gas temperature of the disk.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")                             \
	X("disk_galaxy", "disk_zscale_kpc", "Vertical scale height of the gas disk, in kpc.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")                    \
	X("disk_galaxy", "flux_sphere_radius_kpc", "Radius of the sphere used to measure fluxes, in kpc.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")       \
	X("disk_galaxy", "halo_vphi_expr", "Expression for the halo azimuthal velocity.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")                        \
	X("disk_galaxy", "initial_scalar_density", "Initial density assigned to the passive scalar.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")            \
	X("disk_galaxy", "magnetic_field_microgauss", "Initial magnetic-field strength, in microgauss.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")         \
	X("disk_galaxy", "particle_file", "Input file containing the initial particle population.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")              \
	X("disk_galaxy", "refine_Rmax_kpc", "Maximum cylindrical radius of the refined disk region, in kpc.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")    \
	X("disk_galaxy", "refine_zmax_kpc", "Maximum height of the refined disk region, in kpc.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")                \
	X("disk_galaxy", "vcirc_file", "Input file containing the circular-velocity profile.", "src/problems/DiskGalaxy/testDiskGalaxy.cpp")                   \
	X("dust", "density_floor", "The minimum dust density value allowed in the simulation. Enforced through EnforceLimits.", "src/QuokkaSimulation.hpp")    \
	X("dust", "enable_coefficient_iteration", "If set to 1, iterates state-dependent stopping-time and charge coefficients at both GIRK stages.",          \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "omega_drag_heating", "Controls the fraction of aerodynamic drag dissipation deposited as gas internal energy in the dust source update.",   \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "omega_gyro_residual",                                                                                                                       \
	  "Controls deposition of the gyrofrequency-dependent part of the discrete RK energy residual in `computeDustDragAndLorentz`.",                        \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "picard_alpha_rtol",                                                                                                                         \
	  "Relative convergence tolerance for the reciprocal stopping time \\\\(\\alpha=1/t_{\\mathrm{s}}\\\\) at each GIRK stage. Must be positive.",         \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "picard_charge_atol",                                                                                                                        \
	  "Absolute convergence tolerance for \\\\(\\xi\\\\) at each GIRK stage; used only when the magnetic field is nonzero. Must be positive.",             \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "picard_charge_rtol",                                                                                                                        \
	  "Relative convergence tolerance for \\\\(\\xi\\\\), scaled by its value from the current Picard iterate. Must be positive.",                         \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "picard_max_iterations",                                                                                                                     \
	  "Maximum number of coefficient iterations per dust source update. Must be a positive integer. A nonconverged cell emits a warning and uses the "     \
	  "final iterate.",                                                                                                                                    \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "print_iteration_counts", "If set to 1, prints dust drag or dust drag-plus-Lorentz iteration counts for debugging.",                         \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("dust", "resolved_rk_scheme",                                                                                                                        \
	  "Selects the GIRK coefficients in resolved branch used by the dust source update. Supported values are `TP2025`, `GL4`, and `Midpoint`. At present " \
	  "this only affects `DustSources::computeDustDragAndLorentz`; `DustSources::computeDustDrag` is not affected.",                                       \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "abort_on_fofc_failure",                                                                                                                    \
	  "If set to 1, the code aborts when first-order flux correction fails to yield a physical state (positive density and pressure). This should only "   \
	  "be disabled (0) for debugging.",                                                                                                                    \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "artificial_viscosity_coefficient",                                                                                                         \
	  "This is the linear artificial viscosity coefficient used in the artificial viscosity term added to the flux. This is the same parameter as "        \
	  "defined in the original PPM paper.",                                                                                                                \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "bulk_viscosity", "Constant bulk-viscosity coefficient when the constant-viscosity model is selected.",                                     \
	  "src/QuokkaSimulation.hpp;src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp")                                                           \
	X("hydro", "low_level_debugging_output",                                                                                                               \
	  "If set to 1, turns on low-level debugging output for each RK stage. Warning: this writes an enormous volume of data to disk! This should only be "  \
	  "used for debugging.",                                                                                                                               \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "plm_limiter",                                                                                                                              \
	  "Selects the slope limiter for PLM reconstruction. Options: `minmod`, `sweby`, or `mc`. Only used when `hydro.reconstruction_order = 2`.",           \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "reconstruction_order",                                                                                                                     \
	  "Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise "   \
	  "parabolic; PPM).",                                                                                                                                  \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "rk_integrator_order",                                                                                                                      \
	  "Determines the order of the RK integrator used. Can be set to 1 (Forward Euler) or 2 (RK2-SSP, also known as Heun's method). This should only be "  \
	  "changed for debugging.",                                                                                                                            \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("hydro", "shear_viscosity", "Constant shear-viscosity coefficient when the constant-viscosity model is selected.",                                   \
	  "src/QuokkaSimulation.hpp;src/problems/HydroShearWave/testHydroShearWave.cpp;src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp")        \
	X("hydro", "use_dual_energy",                                                                                                                          \
	  "If set to 1, the code evolves an auxiliary internal energy variable in order to correctly evolve high-mach flows. This should only be disabled "    \
	  "(0) for debugging.",                                                                                                                                \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("jeansRefine", "density_threshold", "Density threshold for Jeans-based refinement in the PopIII problem.", "src/problems/PopIII/testPopIII.cpp")     \
	X("jeansRefine", "ncells", "Target number of cells resolving the Jeans length.", "src/problems/PopIII/testPopIII.cpp")                                 \
	X("marshak", "use_wavespeed_correction", "Enable the wavespeed correction in the asymptotic Marshak-wave test.",                                       \
	  "src/problems/RadMarshakAsymptotic/testRadMarshakAsymptotic.cpp")                                                                                    \
	X("mhd", "emf_averaging_scheme", "Determines the method used to average EMF at edges. Can be set to `LondrilloDelZanna2004` or `Balsara2025`.",        \
	  "src/QuokkaSimulation.hpp;src/simulation.hpp")                                                                                                       \
	X("mhd", "emf_compute_scheme", "Algorithm used to compute edge-centred electromotive forces.", "src/QuokkaSimulation.hpp;src/simulation.hpp")          \
	X("mhd", "emf_reconstruction_order",                                                                                                                   \
	  "Determines the order of spatial reconstruction algorithm used for EMF computation. Can be set to 1 (piecewise constant), 2 (piecewise linear; "     \
	  "PLM), 3 (piecewise parabolic; PPM), or 5 (extrema-preserving xPPM).",                                                                               \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("mhd", "plm_limiter",                                                                                                                                \
	  "Selects the slope limiter for PLM reconstruction in EMF calculations. Options: `minmod`, `sweby`, or `mc`. Only used when "                         \
	  "`mhd.emf_reconstruction_order = 2`.",                                                                                                               \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("mhd", "project_initial_b_field", "If set to 1, projects the initial magnetic field to be divergence-free.", "src/QuokkaSimulation.hpp")             \
	X("mhd", "resistivity", "Constant magnetic resistivity when the constant-resistivity model is selected.",                                              \
	  "src/QuokkaSimulation.hpp;src/problems/MHDAlfvenWaveCircularConvergence/testMHDAlfvenWaveCircularConvergence.cpp;src/problems/"                      \
	  "MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/"      \
	  "problems/MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                    \
	X("mhd", "update_initial_b_energy", "If set to 1, updates the initial magnetic energy after projection.", "src/QuokkaSimulation.hpp")                  \
	X("network", "energy_switch", "Enable energy coupling in the one-zone photoionization network.",                                                       \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("network", "recombination_switch", "Enable recombinations in the Strömgren-sphere network.",                                                         \
	  "src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/testStromgrenSphereRSLA.cpp")                                 \
	X("particles", "SN_p_term_Msunkmps",                                                                                                                   \
	  "Terminal momentum of the supernova remnant in units of \\\\(M_\\odot\\,\\mathrm{km\\,s}^{-1}\\\\). The shell-formation mass "                       \
	  "\\\\(M_\\mathrm{sf}\\\\) is scaled as \\\\((p/p_\\mathrm{canonical})^2\\\\) so that the kinetic energy \\\\(p^2/(2M_\\mathrm{sf})\\\\) is "         \
	  "preserved.",                                                                                                                                        \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "SN_p_term_exponent",                                                                                                                   \
	  "Exponent \\\\(\\alpha_p\\\\) of the ambient-density scaling of the supernova terminal momentum, \\\\(p_{\\mathrm{snr}} = p_{\\mathrm{snr},0} \\, "  \
	  "n_\\mathrm{H}^{\\alpha_p}\\\\).",                                                                                                                   \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "SN_scheme",                                                                                                                            \
	  "Scheme for SN feedback. Options: SN_thermal_only, SN_thermal_or_thermal_momentum, SN_thermal_kinetic_or_thermal_momentum, "                         \
	  "SN_pure_kinetic_or_thermal_momentum.",                                                                                                              \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "SN_smooth_gas_velocity", "Smooth gas velocity in the stencil to enforce energy conservation", "src/particles/particle_types.hpp")      \
	X("particles", "disable_SN_feedback", "If set to 1, disables SN feedback when a particle evolves from SNProgenitor to SNRemnant.",                     \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "disable_particle_drift", "If set to 1, disables particle drift.", "src/particles/particle_types.hpp")                                  \
	X("particles", "eps_ff", "Star formation efficiency parameter.", "src/particles/particle_types.hpp;src/problems/ParticleSF/testParticleSF.cpp")        \
	X("particles", "low_mass_composite_max_mass", "Maximum mass represented by a low-mass composite stellar particle.",                                    \
	  "src/particles/particle_types.hpp;src/problems/ParticleSF/testParticleSF.cpp")                                                                       \
	X("particles", "param1", "Placeholder parameter for particles (used in gravity_3d.cpp tests).", "src/particles/particle_types.hpp")                    \
	X("particles", "param2", "Placeholder parameter for particles (used in gravity_3d.cpp tests).", "src/particles/particle_types.hpp")                    \
	X("particles", "rad_table", "Path to the radiation luminosity table.", "src/simulation.hpp")                                                           \
	X("particles", "rad_table_output_spacing", "Deprecated name for `particles.rad_table_output_transform`.", "src/simulation.hpp")                        \
	X("particles", "rad_table_output_transform",                                                                                                           \
	  "Transform used to interpolate the luminosity table outputs: `linear`, `log`, or `fast_log`. A table containing zeros requires `linear`.",           \
	  "src/problems/ParticleRadiation/testParticleRadiation.cpp;src/simulation.hpp")                                                                       \
	X("particles", "reproducibility_roundoff_redundancy", "Number of bits to remove from the significand for reproducibility.",                            \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "scalar_yield_per_SN", "Passive-scalar yield assigned per supernova.",                                                                  \
	  "src/particles/particle_types.hpp;src/problems/DiskGalaxy/testDiskGalaxy.cpp;src/problems/TallBoxSf/testTallBoxSf.cpp")                              \
	X("particles", "sink_max_alfven_speed", "Maximum post-accretion Alfvén speed in cm/s. A negative value disables the limiter; zero is invalid.",        \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "sink_particle_use_uniform_kernel", "If set to 1, uses uniform accretion kernel in a (7 dx)^3 box for sink particles.",                 \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "split_particles_on_restart_refine", "Whether to split particles when restarting with refinement.", "src/simulation.hpp")               \
	X("particles", "stellar_velocity_limit",                                                                                                               \
	  "Maximum velocity limit for stellar particles in cm/s. The code will abort if stellar velocity exceeds this limit.",                                 \
	  "src/particles/particle_types.hpp")                                                                                                                  \
	X("particles", "use_luminosity_table", "If set to 1, uses a luminosity table for particles.", "src/simulation.hpp")                                    \
	X("particles", "verbose", "Verbosity level for particle operations. Higher values provide more detailed output.", "src/particles/particle_types.hpp")  \
	X("perturb", "cloud_density", "Density of the perturbed cloud in the StarCluster problem.", "src/problems/StarCluster/testStarCluster.cpp")            \
	X("perturb", "cloud_numdens", "Number density of the perturbed cloud in the PopIII problem.", "src/problems/PopIII/testPopIII.cpp")                    \
	X("perturb", "cloud_omega", "Angular velocity imposed on the perturbed cloud.", "src/problems/PopIII/testPopIII.cpp")                                  \
	X("perturb", "cloud_radius", "Radius of the perturbed cloud.", "src/problems/PopIII/testPopIII.cpp;src/problems/StarCluster/testStarCluster.cpp")      \
	X("perturb", "filename", "Input file used to seed the PopIII perturbation.",                                                                           \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/StarCluster/testStarCluster.cpp")                                                                   \
	X("perturb", "rms_velocity", "RMS velocity of the seeded perturbation.", "src/problems/PopIII/testPopIII.cpp")                                         \
	X("perturb", "virial_parameter", "Initial virial parameter of the StarCluster cloud.", "src/problems/StarCluster/testStarCluster.cpp")                 \
	X("photochemistry", "enabled", "Enable photochemistry source terms.", "src/QuokkaSimulation.hpp")                                                      \
	X("photochemistry", "max_density_allowed", "Upper gas-density limit for photochemistry updates.", "src/QuokkaSimulation.hpp")                          \
	X("photochemistry", "min_density_allowed", "Lower gas-density limit for photochemistry updates.", "src/QuokkaSimulation.hpp")                          \
	X("photoionization", "n_HII_init", "Sets the initial ionized-hydrogen number density in the one-zone photoionization test.",                           \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "n_HI_init", "Sets the initial neutral-hydrogen number density in the one-zone photoionization test.",                            \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "n_e_init", "Sets the initial electron number density in the one-zone photoionization test.",                                     \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "n_photon", "Sets the initial photon number density in the one-zone photoionization test.",                                       \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "small_dens", "Sets the minimum density used by the initial state in the one-zone photoionization test.",                         \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "small_temp", "Sets the minimum temperature used by the initial state in the one-zone photoionization test.",                     \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "temperature", "Sets the initial gas temperature in the one-zone photoionization test.",                                          \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization", "tend", "Sets the end time of the test in the one-zone photoionization test.",                                                    \
	  "src/problems/OneZonePhotoionization/testOneZonePhotoionization.cpp")                                                                                \
	X("photoionization_momentum", "n_HII_init", "Sets the initial ionized-hydrogen number density in the photoionization momentum test.",                  \
	  "src/problems/DTypeFrontVC/testDTypeFrontVC.cpp")                                                                                                    \
	X("photoionization_momentum", "n_HI_init", "Sets the initial neutral-hydrogen number density in the photoionization momentum test.",                   \
	  "src/problems/DTypeFrontVC/testDTypeFrontVC.cpp")                                                                                                    \
	X("photoionization_momentum", "n_e_init", "Sets the initial electron number density in the photoionization momentum test.",                            \
	  "src/problems/DTypeFrontVC/testDTypeFrontVC.cpp")                                                                                                    \
	X("photoionization_momentum", "small_dens", "Sets the minimum density used by the initial state in the photoionization momentum test.",                \
	  "src/problems/DTypeFrontVC/testDTypeFrontVC.cpp")                                                                                                    \
	X("photoionization_momentum", "small_temp", "Sets the minimum temperature used by the initial state in the photoionization momentum test.",            \
	  "src/problems/DTypeFrontVC/testDTypeFrontVC.cpp")                                                                                                    \
	X("photoionization_momentum", "temperature", "Sets the initial gas temperature in the photoionization momentum test.",                                 \
	  "src/problems/DTypeFrontVC/testDTypeFrontVC.cpp")                                                                                                    \
	X("photoionize", "T_dust_destroy", "Dust-destruction temperature in the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")         \
	X("photoionize", "beamed", "Inject each wing of the radiation source outward when enabled; inject isotropically when disabled.",                       \
	  "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                                                                                                    \
	X("photoionize", "flux", "Incident radiation flux in the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                        \
	X("photoionize", "flux_ion", "Incident ionizing-photon flux in the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")              \
	X("photoionize", "kappa1", "First opacity coefficient used by the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")               \
	X("photoionize", "kappa2", "Second opacity coefficient used by the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")              \
	X("photoionize", "n_HII_init", "Sets the initial ionized-hydrogen number density in the DTypeFront1D test.",                                           \
	  "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                                                                                                    \
	X("photoionize", "n_HI_init", "Sets the initial neutral-hydrogen number density in the DTypeFront1D test.",                                            \
	  "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                                                                                                    \
	X("photoionize", "n_e_init", "Sets the initial electron number density in the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")   \
	X("photoionize", "small_dens", "Sets the minimum density used by the initial state in the DTypeFront1D test.",                                         \
	  "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                                                                                                    \
	X("photoionize", "small_temp", "Sets the minimum temperature used by the initial state in the DTypeFront1D test.",                                     \
	  "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                                                                                                    \
	X("photoionize", "source_cells", "Number of source cells on each side of the central radiation slab.",                                                 \
	  "src/problems/DTypeFront1D/testDTypeFront1D.cpp")                                                                                                    \
	X("photoionize", "temperature", "Sets the initial gas temperature in the DTypeFront1D test.", "src/problems/DTypeFront1D/testDTypeFront1D.cpp")        \
	X("primordial_chem", "primary_species_1", "Initial abundance of primordial-chemistry primary species 1 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_10", "Initial abundance of primordial-chemistry primary species 10 in the PopIII problem.",                      \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_11", "Initial abundance of primordial-chemistry primary species 11 in the PopIII problem.",                      \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_12", "Initial abundance of primordial-chemistry primary species 12 in the PopIII problem.",                      \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_13", "Initial abundance of primordial-chemistry primary species 13 in the PopIII problem.",                      \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_14", "Initial abundance of primordial-chemistry primary species 14 in the PopIII problem.",                      \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_2", "Initial abundance of primordial-chemistry primary species 2 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_3", "Initial abundance of primordial-chemistry primary species 3 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_4", "Initial abundance of primordial-chemistry primary species 4 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_5", "Initial abundance of primordial-chemistry primary species 5 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_6", "Initial abundance of primordial-chemistry primary species 6 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_7", "Initial abundance of primordial-chemistry primary species 7 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_8", "Initial abundance of primordial-chemistry primary species 8 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "primary_species_9", "Initial abundance of primordial-chemistry primary species 9 in the PopIII problem.",                        \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "small_dens", "Density floor used by the PopIII primordial-chemistry setup.",                                                     \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "small_temp", "Temperature floor used by the PopIII primordial-chemistry setup.",                                                 \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("primordial_chem", "temperature", "Initial gas temperature in the PopIII chemistry setup.",                                                          \
	  "src/problems/PopIII/testPopIII.cpp;src/problems/PrimordialChem/testPrimordialChem.cpp")                                                             \
	X("problem", "IC_file", "Input file containing TallBoxSf initial conditions.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                             \
	X("problem", "M0_in_Msun", "Initial stellar mass for the particle-evolution test, in solar masses.",                                                   \
	  "src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp")                                                                                  \
	X("problem", "SN_particles_file", "Input supernova-particle file for the SN test.", "src/problems/SN/testSN.cpp")                                      \
	X("problem", "T_amb", "Ambient gas temperature in the RandomBlast test.", "src/problems/RandomBlast/testRandomBlast.cpp")                              \
	X("problem", "Tamb", "Ambient gas temperature in the ParticleSF test.", "src/problems/ParticleSF/testParticleSF.cpp")                                  \
	X("problem", "boost_vel_x", "x velocity added to the ParticleSink test setup.",                                                                        \
	  "src/problems/ParticleSink/testParticleSink.cpp;src/problems/SN/testSN.cpp")                                                                         \
	X("problem", "do_split_particles", "Enable particle splitting in the BinaryOrbitCIC test.", "src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp")      \
	X("problem", "flux_source", "Use a flux-based radiation source in the RadStreaming test.", "src/problems/RadStreaming/testRadStreaming.cpp")           \
	X("problem", "history_dt_over_ts0", "Sampling interval for RDI history output, in initial stopping times.",                                            \
	  "src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp")                                                                                          \
	X("problem", "hot_T", "Temperature of the hot phase in the TallBoxSf setup.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                              \
	X("problem", "initial_scalar_density", "Initial passive-scalar density in the TallBoxSf setup.", "src/problems/TallBoxSf/testTallBoxSf.cpp")           \
	X("problem", "kappa1", "First dust opacity coefficient in the Marshak-wave test.",                                                                     \
	  "src/problems/RadMarshakDust/testRadMarshakDust.cpp;src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp")                                         \
	X("problem", "kappa2", "Second dust opacity coefficient in the Marshak-wave test.",                                                                    \
	  "src/problems/RadMarshakDust/testRadMarshakDust.cpp;src/problems/RadMarshakDustPE/testRadMarshakDustPE.cpp")                                         \
	X("problem", "n0", "Initial gas number density in the ParticleSF test.", "src/problems/ParticleSF/testParticleSF.cpp")                                 \
	X("problem", "n_amb", "Ambient gas number density in the RandomBlast test.",                                                                           \
	  "src/problems/RandomBlast/testRandomBlast.cpp;src/problems/SN/testSN.cpp")                                                                           \
	X("problem", "noise_amplitude", "Amplitude of the initial RDI noise perturbation.", "src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp")        \
	X("problem", "noise_seed", "Random seed for the initial RDI noise perturbation.", "src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp")          \
	X("problem", "num_particles", "Number of particles created by the SphericalCollapse test.",                                                            \
	  "src/problems/SphericalCollapse/testSphericalCollapse.cpp")                                                                                          \
	X("problem", "part_fn", "ASCII file used to initialize the RandomBlast stellar particles.", "src/problems/RandomBlast/testRandomBlast.cpp")            \
	X("problem", "particles_file", "Input particle file for the ParticleSink test.", "src/problems/ParticleSink/testParticleSink.cpp")                     \
	X("problem", "particles_filename", "Input particle file for the ParticleRadiation test.",                                                              \
	  "src/problems/ParticleRadiation/testParticleRadiation.cpp;src/problems/ParticleRadiationSlug/testParticleRadiationSlug.cpp")                         \
	X("problem", "reference_steps", "Reference number of steps used by the DustyAlfvenWave test.", "src/problems/DustyAlfvenWave/testDustyAlfvenWave.cpp") \
	X("problem", "refine_center", "Centre of the refined region in the ParticleAccretion test.",                                                           \
	  "src/problems/ParticleAccretion/testParticleAccretion.cpp")                                                                                          \
	X("problem", "refine_half_domain", "Refine half of the domain in the ParticleCreation test.",                                                          \
	  "src/problems/ParticleCreation/testParticleCreation.cpp;src/problems/ParticleSink/testParticleSink.cpp;src/problems/SN/testSN.cpp")                  \
	X("problem", "return_1_at_fail", "Return a failure status when the ParticleAccretion check fails.",                                                    \
	  "src/problems/ParticleAccretion/testParticleAccretion.cpp")                                                                                          \
	X("problem", "rho0", "Initial gas density in the ParticleAccretion test.",                                                                             \
	  "src/problems/ParticleAccretion/testParticleAccretion.cpp;src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp")                         \
	X("problem", "rho01", "Initial density of the first gas phase in TallBoxSf.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                              \
	X("problem", "seed", "Random seed for particle placement in SphericalCollapse.", "src/problems/SphericalCollapse/testSphericalCollapse.cpp")           \
	X("problem", "sigma1", "Velocity dispersion of the first gas phase in TallBoxSf.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                         \
	X("problem", "sink_file", "Input sink-particle file for the ParticleAccretion test.", "src/problems/ParticleAccretion/testParticleAccretion.cpp")      \
	X("problem", "split_factor", "Number of child particles created per split in BinaryOrbitCIC.", "src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp")   \
	X("problem", "star_mass", "Stellar mass used by the ParticleAccretion test.", "src/problems/ParticleAccretion/testParticleAccretion.cpp")              \
	X("problem", "stars_file", "Input stellar-particle file for TallBoxSf.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                                   \
	X("problem", "t_end_over_t_b", "End time expressed in the ParticleAccretion reference timescale.",                                                     \
	  "src/problems/ParticleAccretion/testParticleAccretion.cpp;src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp")                         \
	X("problem", "turnon_fextract", "Enable the fextract diagnostic in ParticleAccretion.", "src/problems/ParticleAccretion/testParticleAccretion.cpp")    \
	X("problem", "uniform_density", "Use a uniform initial gas density in ParticleAccretion.", "src/problems/ParticleAccretion/testParticleAccretion.cpp") \
	X("problem", "validate_initial_imf_stats", "Check the initial stellar initial-mass-function statistics in ParticleSF.",                                \
	  "src/problems/ParticleSF/testParticleSF.cpp")                                                                                                        \
	X("problem", "verify_low_mass_cap_on_restart", "Check the low-mass composite-particle cap after restart in ParticleSF.",                               \
	  "src/problems/ParticleSF/testParticleSF.cpp")                                                                                                        \
	X("problem", "verify_particle_layout", "Check the particle layout in the BinaryOrbitCIC test.", "src/problems/BinaryOrbitCIC/testBinaryOrbitCIC.cpp")  \
	X("problem", "warm_T", "Temperature of the warm phase in TallBoxSf.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                                      \
	X("problem", "write_csv", "Write CSV output from the problem test.",                                                                                   \
	  "src/problems/DustDampedGyromotion/testDustDampedGyromotion.cpp;src/problems/DustDampingMHDZeroBMixedStiff/"                                         \
	  "testDustDampingMHDZeroBMixedStiff.cpp;src/problems/DustLorentzShock/testDustLorentzShock.cpp;src/problems/DustMagnetizedRDI/"                       \
	  "testDustMagnetizedRDI.cpp;src/problems/DustyAlfvenWave/testDustyAlfvenWave.cpp;src/problems/DustyOrszagTang/testDustyOrszagTang.cpp")               \
	X("quokka_time_units", "Gyr", "Number of seconds in one gigayear for parser time expressions.", "src/simulation.hpp")                                  \
	X("quokka_time_units", "Myr", "Number of seconds in one megayear for parser time expressions.", "src/simulation.hpp")                                  \
	X("quokka_time_units", "kyr", "Number of seconds in one kiloyear for parser time expressions.", "src/simulation.hpp")                                  \
	X("quokka_time_units", "yr", "Number of seconds in one year for parser time expressions.", "src/simulation.hpp")                                       \
	X("radiation", "cfl", "Sets the CFL number for the radiation advance. This is independent of the hydro CFL number.", "src/QuokkaSimulation.hpp")       \
	X("radiation", "dust_gas_interaction_coeff", "Coefficient for dust-gas interaction in radiation calculations.", "src/QuokkaSimulation.hpp")            \
	X("radiation", "iteration_tolerance", "Tolerance for the Newton-Raphson iteration residuals.", "src/QuokkaSimulation.hpp")                             \
	X("radiation", "iteration_tolerance_rel", "Tolerance for the relative change between two consecutive Newton-Raphson iterations.",                      \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("radiation", "print_iteration_counts", "If set to 1, prints radiation iteration counts for debugging.", "src/QuokkaSimulation.hpp")                  \
	X("radiation", "reconstruction_order",                                                                                                                 \
	  "Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise "   \
	  "parabolic; PPM).",                                                                                                                                  \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("setup", "advection", "Enable vortex advection in the MHDBalsaraVortex test.", "src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp")             \
	X("setup", "advection_angle_deg", "Angle of the FieldLoop advection velocity, in degrees.", "src/problems/FieldLoop/testFieldLoop.cpp")                \
	X("setup", "advection_vz", "z component of the FieldLoop advection velocity.", "src/problems/FieldLoop/testFieldLoop.cpp")                             \
	X("setup", "angle_between_k_b0", "Angle between the wavevector and background magnetic field, in degrees.",                                            \
	  "src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDEntropyWaveConvergence/"                         \
	  "testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/MHDSlowWaveConvergence/"          \
	  "testMHDSlowWaveConvergence.cpp")                                                                                                                    \
	X("setup", "error_tol", "Error tolerance used by the HydroShearWave test.",                                                                            \
	  "src/problems/HydroShearWave/testHydroShearWave.cpp;src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/"                    \
	  "MHDAlfvenWaveCircularConvergence/testMHDAlfvenWaveCircularConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/"                             \
	  "testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/"                      \
	  "MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                          \
	X("setup", "loop_center_x", "x coordinate of the FieldLoop centre.", "src/problems/FieldLoop/testFieldLoop.cpp")                                       \
	X("setup", "loop_center_y", "y coordinate of the FieldLoop centre.", "src/problems/FieldLoop/testFieldLoop.cpp")                                       \
	X("setup", "loop_radius", "Radius of the magnetic field loop.", "src/problems/FieldLoop/testFieldLoop.cpp")                                            \
	X("setup", "machine_precision_target", "Target accuracy for the HydroWaveConvergence test.",                                                           \
	  "src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/MHDAlfvenWaveCircularConvergence/"                                      \
	  "testMHDAlfvenWaveCircularConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/"          \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                             \
	X("setup", "num_modes_x", "Number of Alfvén-wave modes along x.",                                                                                      \
	  "src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDEntropyWaveConvergence/"                         \
	  "testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/MHDSlowWaveConvergence/"          \
	  "testMHDSlowWaveConvergence.cpp")                                                                                                                    \
	X("setup", "num_modes_y", "Number of Alfvén-wave modes along y.",                                                                                      \
	  "src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDEntropyWaveConvergence/"                         \
	  "testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/MHDSlowWaveConvergence/"          \
	  "testMHDSlowWaveConvergence.cpp")                                                                                                                    \
	X("setup", "num_modes_z", "Number of Alfvén-wave modes along z.",                                                                                      \
	  "src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDEntropyWaveConvergence/"                         \
	  "testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/MHDSlowWaveConvergence/"          \
	  "testMHDSlowWaveConvergence.cpp")                                                                                                                    \
	X("setup", "num_periods", "Number of FieldLoop advection periods to run.",                                                                             \
	  "src/problems/FieldLoop/testFieldLoop.cpp;src/problems/MHDAlfvenWaveCircularConvergence/testMHDAlfvenWaveCircularConvergence.cpp;src/problems/"      \
	  "MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp;src/problems/"         \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                             \
	X("setup", "nx_max", "Largest x resolution in the HydroWaveConvergence sweep.",                                                                        \
	  "src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/MHDAlfvenWaveCircularConvergence/"                                      \
	  "testMHDAlfvenWaveCircularConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/"          \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                             \
	X("setup", "nx_start", "Starting x resolution in the HydroWaveConvergence sweep.",                                                                     \
	  "src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/MHDAlfvenWaveCircularConvergence/"                                      \
	  "testMHDAlfvenWaveCircularConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/"          \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                             \
	X("setup", "refine_based_on", "Field or criterion used to trigger refinement in FieldLoop.", "src/problems/FieldLoop/testFieldLoop.cpp")               \
	X("setup", "refine_n_dims", "Number of dimensions refined in the HydroWaveConvergence test.",                                                          \
	  "src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/"                                        \
	  "testMHDAlfvenWaveLinearConvergence.cpp;src/problems/MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/"                      \
	  "MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                          \
	X("setup", "region_hi_x", "Upper x boundary of the FieldLoop refinement region.", "src/problems/FieldLoop/testFieldLoop.cpp")                          \
	X("setup", "region_hi_y", "Upper y boundary of the FieldLoop refinement region.", "src/problems/FieldLoop/testFieldLoop.cpp")                          \
	X("setup", "region_hi_z", "Upper z boundary of the FieldLoop refinement region.", "src/problems/FieldLoop/testFieldLoop.cpp")                          \
	X("setup", "region_lo_x", "Lower x boundary of the FieldLoop refinement region.", "src/problems/FieldLoop/testFieldLoop.cpp")                          \
	X("setup", "region_lo_y", "Lower y boundary of the FieldLoop refinement region.", "src/problems/FieldLoop/testFieldLoop.cpp")                          \
	X("setup", "region_lo_z", "Lower z boundary of the FieldLoop refinement region.", "src/problems/FieldLoop/testFieldLoop.cpp")                          \
	X("setup", "run_convergence", "Run the HydroWaveConvergence resolution study.",                                                                        \
	  "src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/MHDAlfvenWaveCircularConvergence/"                                      \
	  "testMHDAlfvenWaveCircularConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/"          \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                             \
	X("setup", "run_sim", "Run the HydroWaveConvergence simulation.",                                                                                      \
	  "src/problems/HydroWaveConvergence/testHydroWaveConvergence.cpp;src/problems/MHDAlfvenWaveCircularConvergence/"                                      \
	  "testMHDAlfvenWaveCircularConvergence.cpp;src/problems/MHDAlfvenWaveLinearConvergence/testMHDAlfvenWaveLinearConvergence.cpp;src/problems/"          \
	  "MHDEntropyWaveConvergence/testMHDEntropyWaveConvergence.cpp;src/problems/MHDFastWaveConvergence/testMHDFastWaveConvergence.cpp;src/problems/"       \
	  "MHDSlowWaveConvergence/testMHDSlowWaveConvergence.cpp")                                                                                             \
	X("setup", "seed_b_fraction", "Fractional amplitude of the seed magnetic field in the dynamo test.",                                                   \
	  "src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp")                                                                                      \
	X("setup", "seed_b_wavenumber", "Wavenumber of the seed magnetic field in the dynamo test.",                                                           \
	  "src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp")                                                                                      \
	X("setup", "shear_flow_axis", "Axis of the imposed shear flow in HydroShearWave.", "src/problems/HydroShearWave/testHydroShearWave.cpp")               \
	X("setup", "shear_grad_axis", "Axis along which the HydroShearWave flow varies.", "src/problems/HydroShearWave/testHydroShearWave.cpp")                \
	X("setup", "vortex_Mach", "Mach number of the MHDBalsaraVortex initial state.", "src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp")              \
	X("setup", "vortex_b_magn", "Magnetic-field strength of the MHDBalsaraVortex initial state.",                                                          \
	  "src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp")                                                                                            \
	X("setup", "vortex_radius", "Radius of the MHDBalsaraVortex initial state.", "src/problems/MHDBalsaraVortex/testMHDBalsaraVortex.cpp")                 \
	X("shell_problem", "filename", "Input filename used by the radiation-hydrodynamic shell problem.", "src/problems/RadhydroShell/testRadhydroShell.cpp") \
	X("stromgen", "Q", "Sets the ionizing-photon emission rate in the Strömgren-front test.",                                                              \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "n_HII_init", "Sets the initial ionized-hydrogen number density in the Strömgren-front test.",                                           \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "n_HI_init", "Sets the initial neutral-hydrogen number density in the Strömgren-front test.",                                            \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "n_e_init", "Sets the initial electron number density in the Strömgren-front test.",                                                     \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "small_dens", "Sets the minimum density used by the initial state in the Strömgren-front test.",                                         \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "small_temp", "Sets the minimum temperature used by the initial state in the Strömgren-front test.",                                     \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "temperature", "Sets the initial gas temperature in the Strömgren-front test.",                                                          \
	  "src/problems/DTypeFront_JAFF/testDTypeFront_JAFF.cpp;src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/"        \
	  "testStromgrenSphereRSLA.cpp")                                                                                                                       \
	X("stromgen", "tend", "Sets the end time of the test in the Strömgren-front test.",                                                                    \
	  "src/problems/StromgrenSphere/testStromgrenSphere.cpp;src/problems/StromgrenSphereRSLA/testStromgrenSphereRSLA.cpp")                                 \
	X("turbulence", "ampl_auto_adjust", "If set to 1, enables automatic amplitude adjustment.",                                                            \
	  "src/QuokkaSimulation.hpp;src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp")                                                             \
	X("turbulence", "ampl_factor", "Amplitude adjust factor for forcing field. Can have 1 value or a comma separated list for each component.",            \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "angles_exp",                                                                                                                          \
	  "If spect_form = 2, this sets the number of modes (angles) in k-shell such that it increases as \\\\(k^angles_exp\\\\).",                            \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "enabled",                                                                                                                             \
	  "If set to 1, enables turbulence driving using [chfeder's turbulence driving module](https://github.com/chfeder/turbulence_generator).",             \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "k_driv",                                                                                                                              \
	  "Characteristic driving scale in units of \\\\(2\\pi/length\\\\). This sets the autocorrelation timescale via tau = (length/target_vdisp)/k_driv.",  \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "k_max", "Maximum driving wavenumber in units of \\\\(2\\pi/length\\\\).", "src/QuokkaSimulation.hpp")                                 \
	X("turbulence", "k_min", "Minimum driving wavenumber in units of \\\\(2\\pi/length\\\\).", "src/QuokkaSimulation.hpp")                                 \
	X("turbulence", "length", "Length of turbulent driving box. Can have 1 value or a comma separated list for each component.",                           \
	  "src/QuokkaSimulation.hpp;src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp")                                                             \
	X("turbulence", "nsteps_per_t_turb", "Number of turbulence driving pattern updates per turnover time.", "src/QuokkaSimulation.hpp")                    \
	X("turbulence", "power_law_exp", "If spect_form = 2, this sets the spectral power-law exponent (e.g., -5/3: Kolmogorov; -2: Burgers).",                \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "random_seed", "Random number seed for driving sequence.", "src/QuokkaSimulation.hpp")                                                 \
	X("turbulence", "sol_weight", "Solenoidal weight. Can be 0.0 (compressive driving), 1.0 (solenoidal driving) or 0.5 (natural mixture).",               \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "spect_form", "Spectral form of the driving amplitude. Can be 0 (band, rectangle, constant), 1 (paraboloid) or 2 (power law).",        \
	  "src/QuokkaSimulation.hpp")                                                                                                                          \
	X("turbulence", "target_vdisp", "Target turbulent velocity dispersion.",                                                                               \
	  "src/QuokkaSimulation.hpp;src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp")                                                             \
	X("vortex", "P0", "Initial gas pressure in the NSCBC vortex test.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                                    \
	X("vortex", "Tgas0", "Initial gas temperature in the NSCBC vortex test.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                              \
	X("vortex", "s_inflow", "Passive-scalar value prescribed at the NSCBC vortex inflow.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                 \
	X("vortex", "strength", "Amplitude of the imposed NSCBC vortex.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                                      \
	X("vortex", "u_inflow", "Longitudinal inflow velocity in the NSCBC vortex test.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                      \
	X("vortex", "v_inflow", "Transverse y inflow velocity in the NSCBC vortex test.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                      \
	X("vortex", "w_inflow", "Transverse z inflow velocity in the NSCBC vortex test.", "src/problems/NscbcVortex/testNscbcVortex.cpp")                      \
	X("", "checkpointtime_interval", "The time interval (in simulated time) between checkpoint outputs.", "src/simulation.hpp")                            \
	X("", "constant_dt", "Optional constant timestep. If set, forces the timestamp to be this value.", "src/simulation.hpp")                               \
	X("", "derived_vars", "A list of the names of derived variables that should be included in plotfile outputs.", "src/simulation.hpp")                   \
	X("", "dt_cutoff",                                                                                                                                     \
	  "Timestep drop detector threshold. If the timestep drops below dt_cutoff * current_time, the simulation aborts with an error message. This helps "   \
	  "detect numerical instabilities early.",                                                                                                             \
	  "src/simulation.hpp")                                                                                                                                \
	X("", "initial_dt", "Optional initial timestep.", "src/simulation.hpp")                                                                                \
	X("", "plottime_interval", "The time interval (in simulated time) between plotfile outputs.", "src/simulation.hpp")                                    \
	X("", "sfh_time_interval", "Time interval for updating/writing star formation history.", "src/simulation.hpp")                                         \
	X("*", "center", "Centre coordinates of the plane sampled by a frame diagnostic.", "src/io/DiagFramePlane.cpp")                                        \
	X("*", "deposit_fields", "Particle quantities to deposit onto the grid before producing a derived field.", "src/io/DerivedParticleDeposition.cpp")     \
	X("*", "diagnostics", "Names of the configured diagnostics to run.", "src/simulation.hpp")                                                             \
	X("*", "field_names", "Fields to include in a frame or plotfile diagnostic.",                                                                          \
	  "src/io/DiagFramePlane.cpp;src/io/DiagPlotfile.cpp;src/io/DiagProjectionPlot.cpp")                                                                   \
	X("*", "filters", "Filters applied before writing a diagnostic.", "src/io/DiagBase.cpp")                                                               \
	X("*", "nBins", "Number of bins along a PDF axis.", "src/io/DiagPDF.cpp")                                                                              \
	X("*", "normal", "Normal vector of a sampled frame plane.", "src/io/DiagFramePlane.cpp;src/io/DiagProjectionPlot.cpp")                                 \
	X("*", "particle_types", "Particle species included in particle deposition.", "src/io/DerivedParticleDeposition.cpp")                                  \
	X("*", "particles", "Particle species included in diagnostic plotfiles.",                                                                              \
	  "src/io/DiagFramePlane.cpp;src/io/DiagParticleTxt.cpp;src/io/DiagPlotfile.cpp;src/io/DiagProjectionPlot.cpp")                                        \
	X("*", "range", "Lower and upper bounds of a PDF axis.", "src/io/DiagPDF.cpp")                                                                         \
	X("*", "time_int", "Diagnostic output interval in simulation time.", "src/io/DiagBase.cpp")                                                            \
	X("*", "type", "Selects the diagnostic implementation.", "src/simulation.hpp")                                                                         \
	X("*", "value_greater", "Keep cells whose filtered field exceeds this value.", "src/io/DiagFilter.cpp")                                                \
	X("*", "value_inrange", "Keep cells whose filtered field lies in this range.", "src/io/DiagFilter.cpp")                                                \
	X("*", "value_less", "Keep cells whose filtered field is below this value.", "src/io/DiagFilter.cpp")                                                  \
	X("*", "var_names", "Names of the fields used as PDF axes.", "src/io/DiagPDF.cpp")                                                                     \
	X("geometry", "prob_hi", "Upper physical-domain coordinates used by the MHD dynamo test.",                                                             \
	  "src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp;src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp")                     \
	X("geometry", "prob_lo", "Lower physical-domain coordinates used by the MHD dynamo test.",                                                             \
	  "src/problems/MHDSmallScaleDynamo/testMHDSmallScaleDynamo.cpp;src/problems/ParticleStarEvolution/testParticleStarEvolution.cpp")                     \
	X("problem", "boost_velocity", "Three-component velocity boost applied to the RandomBlast setup.", "src/problems/RandomBlast/testRandomBlast.cpp")     \
	X("problem", "refine_zmax", "Maximum height of the refined region in TallBoxSf.", "src/problems/TallBoxSf/testTallBoxSf.cpp")                          \
	X("problem", "stage_times_over_ts0", "RDI stage times expressed in initial stopping times.",                                                           \
	  "src/problems/DustMagnetizedRDI/testDustMagnetizedRDI.cpp")                                                                                          \
	X("quokka", "bc",                                                                                                                                      \
	  "Boundary conditions for the domain faces. Must be a list of 3 strings (e.g., `periodic periodic reflecting`). Overrides `geometry.is_periodic`.",   \
	  "src/main.cpp;src/simulation.hpp")                                                                                                                   \
	X("turbulence", "stop_time", "Time at which to stop turbulence driving. Default is max, i.e. never stop.", "src/QuokkaSimulation.hpp")                 \
	X("dust", "grain_radius", "Dust grain radius for each group.", "src/dust/DustRuntimeParams.hpp")                                                       \
	X("dust", "grain_density", "Dust grain material density for each group.", "src/dust/DustRuntimeParams.hpp")                                            \
	X("", "list_options", "Print the compiled Quokka option registry and exit.", "src/main.cpp")

namespace quokka
{

struct ParmParseOption {
	std::string_view prefix;
	std::string_view name;
	std::string_view description;
	std::string_view sources;
};

#define QUOKKA_OPTION_ENTRY(prefix, name, description, source) {prefix, name, description, source},
inline constexpr ParmParseOption parmParseOptions[] = {QUOKKA_PARM_PARSE_OPTIONS(QUOKKA_OPTION_ENTRY)};
#undef QUOKKA_OPTION_ENTRY

template <std::size_t N> struct FixedString {
	char value[N]{};
	constexpr FixedString(char const (&text)[N])
	{
		for (std::size_t i = 0; i < N; ++i) {
			value[i] = text[i];
		}
	}
	[[nodiscard]] constexpr auto view() const -> std::string_view { return {value, N - 1}; }
};

template <FixedString Prefix, FixedString Name> consteval auto isRegisteredOption() -> bool
{
	for (auto const &option : parmParseOptions) {
		if (option.prefix == Prefix.view() && option.name == Name.view()) {
			return true;
		}
	}
	return false;
}

inline void printParmParseOptions(std::ostream &out, std::string_view problem)
{
	std::string const problem_source = "src/problems/" + std::string(problem) + "/";
	for (auto const &option : parmParseOptions) {
		bool applies = false;
		std::string_view remaining = option.sources;
		while (!remaining.empty()) {
			auto const separator = remaining.find(';');
			auto const source = remaining.substr(0, separator);
			if (!source.starts_with("src/problems/") || (!problem.empty() && source.starts_with(problem_source))) {
				applies = true;
				break;
			}
			if (separator == std::string_view::npos) {
				break;
			}
			remaining.remove_prefix(separator + 1);
		}
		if (!applies) {
			continue;
		}
		if (!option.prefix.empty()) {
			out << (option.prefix == "*" ? "<diagnostic>" : option.prefix) << '.';
		}
		out << option.name << "\n  " << option.description << "\n";
	}
}

} // namespace quokka

#endif // QUOKKA_PARM_PARSE_OPTION_REGISTRY_HPP_
