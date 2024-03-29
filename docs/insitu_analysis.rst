.. insitu_analysis

In-situ analysis
=====

*In-situ analysis* refers to analyzing the simulations as they are running.
There are two options: using the *runtime diagnostics* that are built-in to Quokka, and using *Ascent*, a third-party library.

Diagnostics
-----------------------
Most of Quokka's diagnostics are adapted from the implementation included in the *Pele* suite of AMReX-based combustion codes.
(See the documentation for `PeleLMeX diagnostics <https://amrex-combustion.github.io/PeleLMeX/manual/html/LMeXControls.html#run-time-diagnostics>`_ for an explanation of the original implementation.)

There are three built-in diagnostics that can be configured to output at periodic intervals while the simulation is running:

* axis-aligned 2D projections
* axis-aligned 2D slices, and 
* N-dimensional probability distribution functions (PDFs).

2D Projections
^^^^^^^^^^^^^^^^^^^^^^^
This diagnostic outputs 2D axis-aligned projections as AMReX plotfiles prefixed with `proj`.

Currently, using this diagnostic requires implementing a custom function in the problem generator for your simulation.
(In the future, this diagnostic may be improved so that it can be configured entirely with runtime parameters.)

The problem generator must call `computePlaneProjection(F const &user_f, const int dir)`
where `user_f` is a lambda function that returns the value to project and `dir` is the axis along which the projection is taken.

*Example problem generator implementation:* ::

  template <> auto RadhydroSimulation<ShockCloud>::ComputeProjections(const int dir) const -> std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>
  {
    // compute density projection
    std::unordered_map<std::string, amrex::BaseFab<amrex::Real>> proj;
    proj["nH"] = computePlaneProjection<amrex::ReduceOpSum>(
        [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::Array4<const Real> const &state) noexcept {
          Real const rho = state(i, j, k, HydroSystem<ShockCloud>::density_index);
          return (quokka::cooling::cloudy_H_mass_fraction * rho) / m_H;
        },
        dir);
    return proj;
  }

*Example input file configuration:* ::

  projection_interval = 200
  projection.dirs = x z

2D Slices
^^^^^^^^^^^^^^^^^^^^^^^
.. note::  This is based on the *DiagFramePlane* diagnostic from PelePhysics, and the same
  runtime parameters should apply here without modification. The output format is also the
  same as that produced by the *Pele* codes.

This outputs 2D slices of the simulation as AMReX plotfiles that can be further examined using, e.g., VisIt or yt.

*Example input file configuration:* ::

  quokka.diagnostics = slice_z           # Space-separated name(s) of diagnostics (arbitrary)
  quokka.slice_z.type = DiagFramePlane   # Diagnostic type (others may be added in the future)
  quokka.slice_z.file = slicez_plt       # Output file prefix (should end in "plt")
  quokka.slice_z.normal = 2              # Plane normal (0 == x, 1 == y, 2 == z)
  quokka.slice_z.center = 2.4688e20      # Coordinate in the normal direction
  quokka.slice_z.int    = 10             # Output interval (in number of coarse steps)
  quokka.slice_z.interpolation = Linear  # Interpolation type: Linear or Quadratic (default: Linear)

  # The problem must output these derived variable(s)
  derived_vars = temperature
  # List of variables to include in output
  quokka.slice_z.field_names = gasDensity gasInternalEnergy temperature

Histograms/PDFs
^^^^^^^^^^^^^^^^^^^^^^^
.. note:: This is based on the *DiagPDF* diagnostic from PelePhysics, but significant changes
  have been made to both the runtime parameters and the output format in order to support
  N-dimensional histograms, log-spaced binning, and mass weighting.

This adds histogram / probability density function (PDF) outputs (as fixed-width text files)
at fixed timestep intervals as the simulation evolves.

Bins can be optionally log-spaced. Normalization of the output is left up to the user,
with bin volumes calculated (for convenience) in both original variable coordinates and optionally-log-transformed coordinates.

*Example input file configuration:* ::

  quokka.hist_temp.type = DiagPDF                         # Diagnostic type
  quokka.hist_temp.file = PDFTempDens                     # Output file prefix
  quokka.hist_temp.int  = 10                              # Output cadence (in number of coarse steps)
  quokka.hist_temp.weight_by = mass                       # (Optional) Weight by: mass, volume, cell_counts
  quokka.hist_temp.var_names = temperature gasDensity     # Variable(s) of interest (compute a N-D histogram)

  quokka.hist_temp.temperature.nBins = 20                 # temperature: Number of bins
  quokka.hist_temp.temperature.log_spaced_bins = 1        # temperature: (Optional, default: 0) Use log-spaced bins
  quokka.hist_temp.temperature.range = 1e3 1e7            # temperature: (Optional) Specify the min/max of the bins

  quokka.hist_temp.gasDensity.nBins = 5                   # gasDensity: Number of bins
  quokka.hist_temp.gasDensity.log_spaced_bins = 1         # gasDensity: (Optional, default: 0) Use log-spaced bins
  quokka.hist_temp.gasDensity.range = 1e-29 1e-23         # gasDensity: (Optional) Specify the min/max of the bins


*Filters (based on any variables, not necessary those used for the histogram) can be optionally added:* ::

  quokka.hist_temp.filters = dense                       # (Optional) List of filters
  quokka.hist_temp.dense.field_name = gasDensity         # Filter field
  quokka.hist_temp.dense.value_greater = 1e-25           # Filters: value_greater, value_less, value_inrange

Ascent (deprecated)
-----------------------
.. warning:: Due to correctness and performance issues, **using Ascent is not recommended**. Support for Ascent will be removed in a future version of Quokka.

Ascent allows you to generate visualizations (as PNG images) while the simulation is running, without any extra effort.

.. note:: On Setonix, Ascent is already built.
  In your job script, add the line:
  ``export Ascent_DIR=/software/projects/pawsey0807/bwibking/ascent_06082023/install/ascent-develop/lib/cmake/ascent``.

Compiling Ascent via Spack
^^^^^^^^^^^^^^^^^^^^^^^
1. Run ``spack external find``.
2. Make sure there are entries listed for ``hdf5``, ``cuda``, and ``openmpi`` in your ``~/.spack/packages.yaml`` file.
3. Add `buildable: False <https://spack.readthedocs.io/en/latest/build_settings.html#external-packages>`_ to each entry.
4. Run ``spack fetch --dependencies ascent@develop+cuda+vtkh~fortran~shared cuda_arch=70 ^conduit~parmetis~fortran``
5. On a dedicated compute node, run ``spack install ascent@develop+cuda+vtkh~fortran~shared cuda_arch=70 ^conduit~parmetis~fortran``

For A100 GPUs, change the above lines to `cuda_arch=80`.
Currently, it's not possible to `build for both GPU models at the same time <https://github.com/Alpine-DAV/ascent/issues/950#issuecomment-1153243232>`_.

Compiling Quokka with Ascent support
^^^^^^^^^^^^^^^^^^^^^^^
1. Load Ascent: ``spack load ascent``
2. Add ``-DAMReX_ASCENT=ON -DAMReX_CONDUIT=ON`` to your CMake options.
3. Compile your problem, e.g.: ``ninja -j4 test_hydro3d_blast``

Customizing the visualization
^^^^^^^^^^^^^^^^^^^^^^^
Add an `ascent_actions.yaml file <https://ascent.readthedocs.io/en/latest/Actions/Actions.html>`_ to the simulation working directory.
This file can even be edited while the simulation is running!

.. warning:: Volume renderings do not correctly handle ghost cells (`GitHub issue <https://github.com/Alpine-DAV/ascent/issues/955>`_).
