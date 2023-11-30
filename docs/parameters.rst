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
     - Selects the method used to interpolate from coarse to fine AMR levels.
   * - stop_time
     - Float
     - The simulation time at which to stop evolving the simulation.
   * - ascent_interval
     - Integer
     - The number of coarse timesteps between Ascent outputs.
   * - plotfile_interval
     - Integer
     - The number of coarse timesteps between plotfile outputs.
   * - projection_interval
     - Integer
     - The number of coarse timesteps between 2D projection outputs.
   * - statistics_interval
     - Integer
     - The number of coarse timesteps between statistics outputs.
   * - checkpointtime_interval
     - Float
     - The time interval (in simulated time) between checkpoint outputs.
   * - checkpoint_interval
     - Float
     - The number of coarse timesteps between checkpoint outputs.
   * - do_reflux
     - Integer
     - this turns on refluxing at coarse-fine boundaries (1) or turns it off (0). Except for debugging, this should always be on if AMR is used.
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
     - The maximum walltime for the simulation in the format DD:HH:SS (days/hours/seconds). After 90% of this walltime elapses, the simulation will stop and exit.
