.. Runtime parameters

Runtime parameters
==========================

This document lists all of the runtime parameters in Quokka.
Using the AMReX ParmParse object, these parameters are read in the `readParameters()` function in `simulation.hpp` from an input file or from command line arguments.

.. list-table:: Table of runtime parameters.
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter Name
     - Type
     - Description
   * - max_timesteps
     - Integer
     - The maximum number of time steps for the simulation.
   * - cfl
     - Float
     - Sets the CFL number for the simulation.
   * - amr_interpolation_method
     - Integer
     - Selects the method (piecewise constant or piecewise linear with limiters) used to interpolate from coarse to fine AMR levels. Except for debugging, this should not be changed.
   * - stop_time
     - Float
     - The simulation time at which to stop evolving the simulation.
   * - ascent_interval
     - Integer
     - The number of coarse timesteps between Ascent outputs.
   * - plotfile_interval
     - Integer
     - The number of coarse timesteps between plotfile outputs.
   * - plottime_interval
     - Float
     - The time interval (in simulated time) between plotfile outputs.
   * - projection_interval
     - Integer
     - The number of coarse timesteps between 2D projection outputs.
   * - statistics_interval
     - Integer
     - The number of coarse timesteps between statistics outputs.
   * - checkpoint_interval
     - Float
     - The number of coarse timesteps between checkpoint outputs.
   * - checkpointtime_interval
     - Float
     - The time interval (in simulated time) between checkpoint outputs.
   * - do_reflux
     - Integer
     - this turns on refluxing at coarse-fine boundaries (1) or turns it off (0). Except for debugging, this should always be on when AMR is used.
   * - suppress_output
     - Integer
     - If set to 1, this disables output to stdout while the simulation is running.
   * - derived_vars
     - String
     - A list of the names of derived variables that should be included in the plotfile and Ascent outputs.
   * - regrid_interval
     - Integer
     - The number of timesteps between AMR regridding.
   * - density_floor
     - Float
     - The minimum density value allowed in the simulation. Enforced through EnforceLimits.
   * - temperature_ceiling
     - Float
     - The ceiling on temperature values in the simulation. Enforced through EnforceLimits.
   * - speed_ceiling
     - Float
     - The ceiling on the absolute value of the fluid velocity in the simulation. Enforced through EnforceLimits.
   * - max_walltime
     - String
     - The maximum walltime for the simulation in the format DD:HH:SS (days/hours/seconds). After 90% of this walltime elapses, the simulation will automatically stop and exit.

Hydrodynamics
^^^^^^^^^^^^^^^^^^^

.. list-table:: Table of hydrodynamics parameters.
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter Name
     - Type
     - Description
   * - hydro.low_level_debugging_output
     - Integer
     - If set to 1, turns on low-level debugging output for each RK stage. Warning: this writes an enormous volume of data to disk! This should only be used for debugging. Default: 0.
   * - hydro.rk_integrator_order
     - Integer
     - Determines the order of the RK integrator used. Can be set to 1 (Forward Euler) or 2 (RK2-SSP, also known as Heun's method). Default: 2. This should only be changed for debugging.
   * - hydro.reconstruction_order
     - Integer
     - Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise parabolic; PPM). Default: 3 (PPM).
   * - hydro.use_dual_energy
     - Integer
     - If set to 1, the code evolves an auxiliary internal energy variable in order to correctly evolve high-mach flows. This should only be disabled (0) for debugging. Default: 1.
   * - hydro.abort_on_fofc_failure
     - Integer
     - If set to 1, the code aborts when first-order flux correction fails to yield a physical state (positive density and pressure). This should only be disabled (0) for debugging.
   * - hydro.artificial_viscosity_coefficient
     - Float
     - This is the linear artificial viscosity coefficient used in the artificial viscosity term added to the flux. This is the same parameter as defined in the original PPM paper. Default: 0.

Radiation
^^^^^^^^^^^^^^^^^^^

.. list-table:: Table of radiation parameters.
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter Name
     - Type
     - Description
   * - radiation.reconstruction_order
     - Integer
     - Determines the order of spatial reconstruction algorithm used. Can be set to 1 (piecewise constant), 2 (piecewise linear; PLM), or 3 (piecewise parabolic; PPM). Default: 3 (PPM).
   * - radiation.cfl
     - Float
     - Sets the CFL number for the radiation advance. This is independent of the hydro CFL number.

Optically-thin radiative cooling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Table of optically-thin radiative cooling parameters.
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter Name
     - Type
     - Description
   * - cooling.enabled
     - Integer
     - If set to 1, turns on optically-thin radiative cooling as a Strang-split source term. Default: 0 (disabled).
   * - cooling.read_tables_even_if_disabled
     - Integer
     - If set to 1, reads the cooling tables even if the cooling module is disabled.
   * - cooling.grackle_data_file
     - String
     - The path to the cooling tables in Grackle-compatible HDF5 format.
