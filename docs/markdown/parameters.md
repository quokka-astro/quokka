# Runtime parameters

This document lists all of the runtime parameters in Quokka that are set using the AMReX ParmParse object. Users can set these via the input file or by using command-line arguments.

## General

These parameters are read in the ``AMRSimulation<problem_t>::readParameters()`` function in ``src/simulation.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| max_timesteps | Integer | The maximum number of time steps for the simulation. |
| cfl | Float | Sets the CFL number for the simulation. |
| amr_interpolation_method | Integer | Selects the method (piecewise constant or piecewise linear with limiters) used to interpolate from coarse to fine AMR levels. Except for debugging, this should not be changed. |
| stop_time | Float | The simulation time at which to stop evolving the simulation. |
| ascent_interval | Integer | The number of coarse timesteps between Ascent outputs. |
| plotfile_interval | Integer | The number of coarse timesteps between plotfile outputs. |
| plottime_interval | Float | The time interval (in simulated time) between plotfile outputs. |
| skip_initial_plotfile | Boolean (0/1) | Skip writing the initial plotfile at t=0. Default: 0 (false). |
| projection_interval | Integer | The number of coarse timesteps between 2D projection outputs. |
| statistics_interval | Integer | The number of coarse timesteps between statistics outputs. |
| checkpoint_interval | Float | The number of coarse timesteps between checkpoint outputs. |
| checkpointtime_interval | Float | The time interval (in simulated time) between checkpoint outputs. |
| do_reflux | Integer | This turns on refluxing at coarse-fine boundaries (1) or turns it off (0). Except for debugging, this should always be on when AMR is used. |
| do_tracers | Integer | This turns on tracer particles. They are initialized one-per-cell and they follow the fluid velocity. Default: 0 (off). |
| suppress_output | Integer | If set to 1, this disables output to stdout while the simulation is running. |
| derived_vars | String | A list of the names of derived variables that should be included in the plotfile and Ascent outputs. |
| regrid_interval | Integer | The number of timesteps between AMR regridding. |
| density_floor | Float | The minimum density value allowed in the simulation. Enforced through EnforceLimits. |
| temperature_floor | Float | The minimum temperature value allowed in the simulation. Enforced through EnforceLimits. |
| max_walltime | String | The maximum walltime for the simulation in the format DD:HH:SS (days/hours/seconds). After 90% of this walltime elapses, the simulation will automatically stop and exit. |
| dt_cutoff | Float | Timestep drop detector threshold. If the timestep drops below dt_cutoff * current_time, the simulation aborts with an error message. This helps detect numerical instabilities early. Default: 0.0 (disabled). |
| particle_cfl | Float | Sets the CFL number for particle advection. This is independent of the hydro CFL number. |
| plotfile_prefix | String | The prefix for plotfile output filenames. Default: "plt". |
| checkpoint_prefix | String | The prefix for checkpoint output filenames. Default: "chk". |
| do_subcycle | Integer | This turns on subcycling at coarse-fine boundaries (1) or turns it off (0). Default: 1 (on). |
| poisson_supercycle_interval | Integer | The number of coarse timesteps between Poisson supercycle operations. |
| poisson_reltol | Float | Relative tolerance for the Poisson solver convergence. Default: 1.0e-5. |
| poisson_abstol | Float | Absolute tolerance for the Poisson solver convergence (scaled by minimum RHS value). Default: 1.0e-5. |
| print_cycle_timing | Integer | If set to 1, prints per-cycle timing information. Default: 0 (disabled). |
| restartfile | String | The path to a checkpoint file from which to restart the simulation. |
| amr.plot_nfiles | Integer | Maximum number of binary files per multifab for plotfiles. Controls parallel I/O chunking. |
| amr.checkpoint_nfiles | Integer | Maximum number of binary files per multifab for checkpoints. Controls parallel I/O chunking. |

## Hydrodynamics

These parameters are read in the ``QuokkaSimulation<problem_t>::readParmParse()`` function in ``src/QuokkaSimulation.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| hydro.low_level_debugging_output | Integer | If set to 1, turns on low-level debugging output for each RK stage. Warning: this writes an enormous volume of data to disk! This should only be used for debugging. Default: 0. |
| hydro.rk_integrator_order | Integer | Determines the order of the RK integrator used. Can be set to 1 (Forward Euler) or 2 (RK2-SSP, also known as Heun's method). Default: 2. This should only be changed for debugging. |
| hydro.reconstruction_order | Integer | Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise parabolic; PPM). Default: 3 (PPM). |
| hydro.use_dual_energy | Integer | If set to 1, the code evolves an auxiliary internal energy variable in order to correctly evolve high-mach flows. This should only be disabled (0) for debugging. Default: 1. |
| hydro.abort_on_fofc_failure | Integer | If set to 1, the code aborts when first-order flux correction fails to yield a physical state (positive density and pressure). This should only be disabled (0) for debugging. |
| hydro.artificial_viscosity_coefficient | Float | This is the linear artificial viscosity coefficient used in the artificial viscosity term added to the flux. This is the same parameter as defined in the original PPM paper. Default: 0. |

## Radiation

These parameters are read in the ``QuokkaSimulation<problem_t>::readParmParse()`` function in ``src/QuokkaSimulation.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| radiation.reconstruction_order | Integer | Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise parabolic; PPM). Default: 3 (PPM). |
| radiation.cfl | Float | Sets the CFL number for the radiation advance. This is independent of the hydro CFL number. |
| radiation.dust_gas_interaction_coeff | Float | Coefficient for dust-gas interaction in radiation calculations. |
| radiation.print_iteration_counts | Integer | If set to 1, prints radiation iteration counts for debugging. Default: 0 (disabled). |

## MHD

These parameters are read in the ``QuokkaSimulation<problem_t>::readParmParse()`` function in ``src/QuokkaSimulation.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| mhd.emf_averaging_method | String | Determines the method used to average EMF at edges. Can be set to `BalsaraSpicer` or `LD04`. Default: `LD04`. |
| mhd.emf_reconstruction_order | Integer | Determines the order of spatial reconstruction algorithm used for EMF computation. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), 3 (piecewise parabolic; PPM), or 5 (extrema-preserving xPPM). Default: 5 (xPPM). |

## Optically-thin radiative cooling

These parameters are read in the ``QuokkaSimulation<problem_t>::readParmParse()`` function in ``src/QuokkaSimulation.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| cooling.enabled | Integer | If set to 1, turns on optically-thin radiative cooling as a Strang-split source term. Default: 0 (disabled). |
| cooling.cooling_table_type | String | Specifies the type of cooling table to use. The only supported option is "resampled". |
| cooling.read_tables_even_if_disabled | Integer | If set to 1, reads the cooling tables even if the cooling module is disabled. |
| cooling.hdf5_data_file | String | The path to the cooling tables in HDF5 format. |

## Chemistry

These parameters are read in the ``QuokkaSimulation<problem_t>::readParmParse()`` function in ``src/QuokkaSimulation.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| chemistry.enabled | Integer | If set to 1, turns on chemistry as a Strang-split source term. Default: 0 (disabled). |
| chemistry.max_density_allowed | Float | Maximum density value for which chemistry calculations are accurate. Chemistry is not performed for cells with densities above this threshold. |
| chemistry.min_density_allowed | Float | Minimum density value for which chemistry calculations are performed. Chemistry is not performed for cells with densities below this threshold. |

## Particles

These parameters are read in the ``particleParmParse()`` function in ``src/particles/particle_types.hpp``.

| Parameter Name | Type | Description |
|----|----|----|
| particles.disable_SN_feedback | Integer | If set to 1, disables SN feedback when a particle evolves from SNProgenitor to SNRemnant. Default: 0 (enabled). |
| particles.sink_particle_use_uniform_kernel | Integer | If set to 1, uses uniform accretion kernel in a (7 dx)^3 box for sink particles. Default: 0 (disabled). |
| particles.SN_scheme | string | Scheme for SN feedback. Options: SN_thermal_only, SN_thermal_or_thermal_momentum, SN_thermal_kinetic_or_thermal_momentum, SN_pure_kinetic_or_thermal_momentum. Default: SN_thermal_or_thermal_momentum. |
| particles.eps_ff | Float | Star formation efficiency parameter. Default: 0.01. |
| particles.verbose | Integer | Verbosity level for particle operations. Higher values provide more detailed output. Default: 0. |
| particles.param1 | Float | Placeholder parameter for particles (used in gravity_3d.cpp tests). Default: -1.0. |
| particles.param2 | Float | Placeholder parameter for particles (used in gravity_3d.cpp tests). Default: -1.0. |
