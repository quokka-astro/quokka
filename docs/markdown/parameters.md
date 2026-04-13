# Runtime parameters

This document lists all of the runtime parameters in Quokka that are set using the AMReX ParmParse object. Users can set these via the input file or by using command-line arguments.

## General

These parameters are read in the `AMRSimulation<problem_t>::readParameters()` function in `src/simulation.hpp` or `src/main.cpp`.

| Parameter Name              | Type          | Default           | Description                                                                                                                                                                           |
|-----------------------------|---------------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| max_timesteps               | Integer       | `INT_MAX`         | The maximum number of time steps for the simulation.                                                                                                                                  |
| cfl                         | Float         | `0.3`             | Sets the CFL number for the simulation.                                                                                                                                               |
| amr_interpolation_method    | Integer       | `1`               | Selects the method used to interpolate from coarse to fine AMR levels. 0: piecewise constant, 1: linear. Except for debugging, this should not be changed.                            |
| stop_time                   | Float         | `1.0`             | The simulation time at which to stop evolving the simulation.                                                                                                                         |
| ascent_interval             | Integer       | `-1` (Disabled)   | The number of coarse timesteps between Ascent outputs.                                                                                                                                |
| plotfile_interval           | Integer       | `-1` (Disabled)   | The number of coarse timesteps between plotfile outputs.                                                                                                                              |
| plottime_interval           | Float         | `-1.0` (Disabled) | The time interval (in simulated time) between plotfile outputs.                                                                                                                       |
| skip_initial_plotfile       | Boolean (0/1) | `0` (False)       | Skip writing the initial plotfile at t=0.                                                                                                                                             |
| statistics_interval         | Integer       | `-1` (Disabled)   | The number of coarse timesteps between statistics outputs.                                                                                                                            |
| checkpoint_interval         | Integer       | `-1` (Disabled)   | The number of coarse timesteps between checkpoint outputs.                                                                                                                            |
| checkpointtime_interval     | Float         | `-1.0` (Disabled) | The time interval (in simulated time) between checkpoint outputs.                                                                                                                     |
| do_reflux                   | Boolean (0/1) | `1` (Enabled)     | This turns on refluxing at coarse-fine boundaries (1) or turns it off (0). Except for debugging, this should always be on when AMR is used.                                           |
| do_tracers                  | Boolean (0/1) | `0` (Disabled)    | This turns on tracer particles. They are initialized one-per-cell and they follow the fluid velocity.                                                                                 |
| suppress_output             | Boolean (0/1) | `0` (Disabled)    | If set to 1, this disables output to stdout while the simulation is running.                                                                                                          |
| derived_vars                | String list   | Empty             | A list of the names of derived variables that should be included in the plotfile and Ascent outputs.                                                                                  |
| regrid_interval             | Integer       | `2`               | The number of timesteps between AMR regridding.                                                                                                                                       |
| density_floor               | Float         | `0.0`             | The minimum density value allowed in the simulation. Enforced through EnforceLimits.                                                                                                  |
| density_floor_expr          | String        | Empty             | Optional AMReX parser expression for a spatially varying density floor. Variables: x, y, z, base_density_floor. When set, this overrides the constant floor.                          |
| debug_density_floor_plot    | Boolean (0/1) | `0` (Disabled)    | If set to 1, adds a derived field `density_floor_dbg` to plotfiles to visualize the spatially varying density floor.                                                                  |
| temperature_floor           | Float         | `0.0`             | The minimum temperature value allowed in the simulation. Enforced through EnforceLimits.                                                                                              |
| max_walltime                | String        | `0` (Unlimited)   | The maximum walltime for the simulation in the format DD:HH:SS (days/hours/seconds). After 90% of this walltime elapses, the simulation will automatically stop and exit.             |
| dt_cutoff                   | Float         | `0.0` (Disabled)  | Timestep drop detector threshold. If the timestep drops below dt_cutoff * current_time, the simulation aborts with an error message. This helps detect numerical instabilities early. |
| constant_dt                 | Float         | `0.0` (Disabled)  | Optional constant timestep. If set, forces the timestamp to be this value.                                                                                                            |
| initial_dt                  | Float         | Unlimited         | Optional initial timestep.                                                                                                                                                            |
| max_dt                      | Float         | Unlimited         | Optional maximum timestep.                                                                                                                                                            |
| init_shrink                 | Float         | `1.0`             | Factor to shrink the initial timestep by.                                                                                                                                             |
| show_performance_hints      | Boolean (0/1) | `1` (Enabled)     | If set to 1, prints performance hints.                                                                                                                                                |
| signal_speed_abort          | Float         | `-1.0` (Disabled) | If value > 0 and the max signal speed exceeds this value, the simulation aborts.                                                                                                      |
| particle_speed_abort        | Float         | `-1.0` (Disabled) | If value > 0 and the max particle speed exceeds this value, the simulation aborts.                                                                                                    |
| sfh_interval                | Integer       | `-1` (Disabled)   | Interval for updating/writing star formation history.                                                                                                                                 |
| sfh_time_interval           | Float         | `-1.0` (Disabled) | Time interval for updating/writing star formation history.                                                                                                                            |
| particle_cfl                | Float         | `0.5`             | Sets the CFL number for particle advection. This is independent of the hydro CFL number.                                                                                              |
| plotfile_prefix             | String        | `"plt"`           | The prefix for plotfile output filenames.                                                                                                                                             |
| checkpoint_prefix           | String        | `"chk"`           | The prefix for checkpoint output filenames.                                                                                                                                           |
| do_subcycle                 | Boolean (0/1) | `1` (Enabled)     | This turns on subcycling at coarse-fine boundaries (1) or turns it off (0).                                                                                                           |
| poisson_supercycle_interval | Integer       | `1`               | The number of coarse timesteps between Poisson supercycle operations.                                                                                                                 |
| poisson_reltol              | Float         | `1.0e-5`          | Relative tolerance for the Poisson solver convergence.                                                                                                                                |
| poisson_abstol              | Float         | `1.0e-5`          | Absolute tolerance for the Poisson solver convergence (scaled by minimum RHS value).                                                                                                  |
| print_cycle_timing          | Boolean (0/1) | `0` (Disabled)    | If set to 1, prints per-cycle timing information.                                                                                                                                     |
| restartfile                 | String        | Empty             | The path to a checkpoint file from which to restart the simulation.                                                                                                                   |
| amr.plot_nfiles             | Integer       | `-1` (Auto)       | Maximum number of binary files per multifab for plotfiles. Controls parallel I/O chunking.                                                                                            |
| amr.checkpoint_nfiles       | Integer       | `-1` (Auto)       | Maximum number of binary files per multifab for checkpoints. Controls parallel I/O chunking.                                                                                          |
| quokka.bc                   | String list   | **Required**      | Boundary conditions for the domain faces. Must be a list of 3 strings (e.g., `periodic periodic reflecting`). Overrides `geometry.is_periodic`.                                       |

## Hydrodynamics

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name                         | Type          | Default        | Description                                                                                                                                                                    |
|----------------------------------------|---------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| hydro.low_level_debugging_output       | Boolean (0/1) | `0` (Disabled) | If set to 1, turns on low-level debugging output for each RK stage. Warning: this writes an enormous volume of data to disk! This should only be used for debugging.           |
| hydro.rk_integrator_order              | Integer       | `2` (RK2-SSP)  | Determines the order of the RK integrator used. Can be set to 1 (Forward Euler) or 2 (RK2-SSP, also known as Heun's method). This should only be changed for debugging.        |
| hydro.reconstruction_order             | Integer       | `3` (PPM)      | Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise parabolic; PPM).               |
| hydro.plm_limiter                      | String        | `"sweby"`      | Selects the slope limiter for PLM reconstruction. Options: `minmod`, `sweby`, or `mc`. Only used when `hydro.reconstruction_order = 2`.                                        |
| hydro.use_dual_energy                  | Boolean (0/1) | `1` (Enabled)  | If set to 1, the code evolves an auxiliary internal energy variable in order to correctly evolve high-mach flows. This should only be disabled (0) for debugging.              |
| hydro.abort_on_fofc_failure            | Boolean (0/1) | `1` (Enabled)  | If set to 1, the code aborts when first-order flux correction fails to yield a physical state (positive density and pressure). This should only be disabled (0) for debugging. |
| hydro.artificial_viscosity_coefficient | Float         | `0.0`          | This is the linear artificial viscosity coefficient used in the artificial viscosity term added to the flux. This is the same parameter as defined in the original PPM paper.  |

## Radiation

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name                       | Type          | Default           | Description                                                                                                                                                      |
|--------------------------------------|---------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| radiation.reconstruction_order       | Integer       | `3` (PPM)         | Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise parabolic; PPM). |
| radiation.cfl                        | Float         | `0.3`             | Sets the CFL number for the radiation advance. This is independent of the hydro CFL number.                                                                      |
| radiation.dust_gas_interaction_coeff | Float         | `2.5e-34`         | Coefficient for dust-gas interaction in radiation calculations.                                                                                                  |
| radiation.print_iteration_counts     | Boolean (0/1) | `0` (Disabled)    | If set to 1, prints radiation iteration counts for debugging.                                                                                                    |
| radiation.iteration_tolerance        | Float         | `1e-11`           | Tolerance for the Newton-Raphson iteration residuals.                                                                                                            |
| radiation.iteration_tolerance_rel    | Float         | `-1.0` (Disabled) | Tolerance for the relative change between two consecutive Newton-Raphson iterations.                                                                             |

## MHD

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name               | Type          | Default                   | Description                                                                                                                                                                                                       |
|------------------------------|---------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mhd.emf_computing_scheme     | String        | `"FelkerStone2017"`       | Determines the method used to compute the EMF at edges. Can be set to `FelkerStone2017`, `Balsara2025`, or `Quokka2026`.                                                                                          |
| mhd.emf_averaging_scheme     | String        | `"LondrilloDelZanna2004"` | Determines the method used to average EMF at edges. Can be set to `LondrilloDelZanna2004` or `Balsara2025`.                                                                                                         |
| mhd.emf_reconstruction_order | Integer       | `5` (xPPM)                | Determines the order of spatial reconstruction algorithm used for EMF computation. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), 3 (piecewise parabolic; PPM), or 5 (extrema-preserving xPPM). |
| mhd.plm_limiter              | String        | `"sweby"`                 | Selects the slope limiter for PLM reconstruction in EMF calculations. Options: `minmod`, `sweby`, or `mc`. Only used when `mhd.emf_reconstruction_order = 2`.                                                     |
| mhd.project_initial_b_field  | Boolean (0/1) | `0` (Disabled)            | If set to 1, projects the initial magnetic field to be divergence-free.                                                                                                                                           |
| mhd.update_initial_b_energy  | Boolean (0/1) | `1` (Enabled)             | If set to 1, updates the initial magnetic energy after projection.                                                                                                                                                |

## Optically-thin radiative cooling

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name                       | Type          | Default                             | Description                                                                                                                                         |
|--------------------------------------|---------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| cooling.enabled                      | Boolean (0/1) | `0` (Disabled)                      | If set to 1, turns on optically-thin radiative cooling as a Strang-split source term.                                                               |
| cooling.cooling_table_type           | String        | `"resampled"`                       | Specifies the type of cooling table to use. The only supported option is "resampled".                                                               |
| cooling.read_tables_even_if_disabled | Boolean (0/1) | `0` (Disabled)                      | If set to 1, reads the cooling tables even if the cooling module is disabled.                                                                       |
| cooling.hdf5_data_file               | String        | **Required** if `cooling.enabled=1` | The path to the cooling tables in HDF5 format. We recommend using `extern/cooling/CloudyData_UVB=HM2012_resampled.h5` for ISM at solar metallicity. |

## Chemistry

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name                | Type          | Default                 | Description                                                                                                                                     |
|-------------------------------|---------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| chemistry.enabled             | Boolean (0/1) | `0` (Disabled)          | If set to 1, turns on chemistry as a Strang-split source term.                                                                                  |
| chemistry.max_density_allowed | Float         | `1.0e300`               | Maximum density value for which chemistry calculations are accurate. Chemistry is not performed for cells with densities above this threshold.  |
| chemistry.min_density_allowed | Float         | Smallest positive Value | Minimum density value for which chemistry calculations are performed. Chemistry is not performed for cells with densities below this threshold. |

## Dust

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name              | Type          | Default        | Description                                                                                                                    |
|-----------------------------|---------------|----------------|--------------------------------------------------------------------------------------------------------------------------------|
| dust.enable_iter_stoptime   | Boolean (0/1) | `0` (Disabled) | If set to 1, enables iterative dust stopping time calculation.                                                                 |
| dust.omega                  | Float         | `1.0`          | Controls the level of frictional heating, with omega = 0 turning it off and omega = 1 depositing all dissipation into the gas. |
| dust.print_iteration_counts | Boolean (0/1) | `0` (Disabled) | If set to 1, prints dust drag iteration counts for debugging.                                                                  |
| dust.density_floor | Float | `0.0` | The minimum dust density value allowed in the simulation. Enforced through EnforceLimits.                                                           |
## Particles

These parameters are read in the `particleParmParse()` function in `src/particles/particle_types.hpp` and `readParmParse()` in `src/simulation.hpp`.

| Parameter Name                                | Type          | Default                                                                                                       | Description                                                                                                                                                    |
|-----------------------------------------------|---------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| particles.disable_SN_feedback                 | Boolean (0/1) | `0`                                                                                                           | If set to 1, disables SN feedback when a particle evolves from SNProgenitor to SNRemnant.                                                                      |
| particles.sink_particle_use_uniform_kernel    | Boolean (0/1) | `0` (Disabled)                                                                                                | If set to 1, uses uniform accretion kernel in a (7 dx)^3 box for sink particles.                                                                               |
| particles.SN_scheme                           | String        | `SN_thermal_or_thermal_momentum`                                                                              | Scheme for SN feedback. Options: SN_thermal_only, SN_thermal_or_thermal_momentum, SN_thermal_kinetic_or_thermal_momentum, SN_pure_kinetic_or_thermal_momentum. |
| particles.SN_p_term_Msunkmps                  | Float         | `2.8e5`                                                                                                       | Terminal momentum of the supernova remnant in units of $M_\odot\,\mathrm{km\,s}^{-1}$. The shell-formation mass $M_\mathrm{sf}$ is scaled as $(p/p_\mathrm{canonical})^2$ so that the kinetic energy $p^2/(2M_\mathrm{sf})$ is preserved. |
| particles.eps_ff                              | Float         | `0.01`                                                                                                        | Star formation efficiency parameter.                                                                                                                           |
| particles.verbose                             | Boolean (0/1) | `0`                                                                                                           | Verbosity level for particle operations. Higher values provide more detailed output.                                                                           |
| particles.param1                              | Float         | `-1.0`                                                                                                        | Placeholder parameter for particles (used in gravity_3d.cpp tests).                                                                                            |
| particles.param2                              | Float         | `-1.0`                                                                                                        | Placeholder parameter for particles (used in gravity_3d.cpp tests).                                                                                            |
| particles.disable_particle_drift              | Boolean (0/1) | `0`                                                                                                           | If set to 1, disables particle drift.                                                                                                                          |
| particles.stellar_velocity_limit              | Float         | `1.0e8`                                                                                                       | Maximum velocity limit for stellar particles in cm/s. The code will abort if stellar velocity exceeds this limit.                                              |
| particles.reproducibility_roundoff_redundancy | Integer       | `20`                                                                                                          | Number of bits to remove from the significand for reproducibility.                                                                                             |
| particles.use_luminosity_table                | Boolean (0/1) | `1` (Enabled)                                                                                                 | If set to 1, uses a luminosity table for particles.                                                                                                            |
| particles.rad_table                           | String        | **Required** if `particles.use_luminosity_table=1` and `Physics_Traits<problem_t>::is_radiation_enabled=true` | Path to the radiation luminosity table.                                                                                                                        |
| particles.rad_table_output_spacing            | Integer       | `0` (fast_log)                                                                                                | Output spacing for radiation table.                                                                                                                            |
| particles.split_particles_on_restart_refine   | Boolean (0/1) | `1` (Enabled)                                                                                                 | Whether to split particles when restarting with refinement.                                                                                                    |

## Turbulence

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`. Details about the turbulence driving implementation can be found [here](https://github.com/chfeder/turbulence_generator).

| Parameter Name               | Type          | Default                                | Description                                                                                                                           |
|------------------------------|---------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| turbulence.enabled           | Boolean (0/1) | `0` (Disabled)                         | If set to 1, enables turbulence driving using [chfeder's turbulence driving module](https://github.com/chfeder/turbulence_generator). |
| turbulence.length            | Float         | **Required** if `turbulence.enabled=1` | Length of turbulent driving box. Can have 1 value or a comma separated list for each component.                                       |
| turbulence.target_vdisp      | Float         | **Required** if `turbulence.enabled=1` | Target turbulent velocity dispersion.                                                                                                 |
| turbulence.ampl_factor       | Float         | **Required** if `turbulence.enabled=1` | Amplitude adjust factor for forcing field. Can have 1 value or a comma separated list for each component.                             |
| turbulence.ampl_auto_adjust  | Boolean (0/1) | `0` (Disabled)                         | If set to 1, enables automatic amplitude adjustment.                                                                                  |
| turbulence.k_driv            | Float         | **Required** if `turbulence.enabled=1` | Characteristic driving scale in units of $2\pi/length$. This sets the autocorrelation timescale via tau = (length/target_vdisp)/k_driv. |
| turbulence.k_min             | Float         | **Required** if `turbulence.enabled=1` | Minimum driving wavenumber in units of $2\pi/length$.                                                                                 |
| turbulence.k_max             | Float         | **Required** if `turbulence.enabled=1` | Maximum driving wavenumber in units of $2\pi/length$.                                                                                 |
| turbulence.sol_weight        | Float         | **Required** if `turbulence.enabled=1` | Solenoidal weight. Can be 0.0 (compressive driving), 1.0 (solenoidal driving) or 0.5 (natural mixture).                               |
| turbulence.spect_form        | Integer       | **Required** if `turbulence.enabled=1` | Spectral form of the driving amplitude. Can be 0 (band, rectangle, constant), 1 (paraboloid) or 2 (power law).                        |
| turbulence.power_law_exp     | Float         | **Required** if `turbulence.spect_form=2` | If spect_form = 2, this sets the spectral power-law exponent (e.g., -5/3: Kolmogorov; -2: Burgers).                                   |
| turbulence.angles_exp        | Float         | **Required** if `turbulence.spect_form=2` | If spect_form = 2, this sets the number of modes (angles) in k-shell such that it increases as $k^angles_exp$.                        |
| turbulence.stop_time         | Float         | **Optional** if `turbulence.enabled=1` | Time at which to stop turbulence driving. Default is max, i.e. never stop.                                                                                              |
| turbulence.random_seed       | Integer       | **Required** if `turbulence.enabled=1` | Random number seed for driving sequence.                                                                                              |
| turbulence.nsteps_per_t_turb | Integer       | **Required** if `turbulence.enabled=1` | Number of turbulence driving pattern updates per turnover time.                                                                       |

## Photoelectric Heating

These parameters are read in the `QuokkaSimulation<problem_t>::readParmParse()` function in `src/QuokkaSimulation.hpp`.

| Parameter Name                   | Type          | Default                                                                                   | Description                                                                                     |
|----------------------------------|---------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| use_sfh_based_pe_heating         | Boolean (0/1) | `0` (Disabled)                                                                            | If set to 1, enables photoelectric heating based on star formation history.                     |
| sfh_to_pe_heating_table          | String        | **Required** if `use_sfh_based_pe_heating = 1` | Path to the table converting star formation history to photoelectric heating.                   |
| sf_area_kpc2                     | Float         | **Required** if `use_sfh_based_pe_heating = 1` and `const_sfr_Msun_per_year_per_kpc2 < 0` | Area of the star formation region in kpc^2.                                                     |
| const_sfr_Msun_per_year_per_kpc2 | Float         | `-1.0`                                                                                    | Constant star formation rate in Msun/year/kpc^2. If non-negative, overrides the simulation SFR. |
