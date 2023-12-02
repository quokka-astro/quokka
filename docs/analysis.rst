.. Analysis

Analyzing simulations
=====

There are several ways to analyze Quokka simulations. One method (Ascent) allows you to run the analysis in memory as the simulation is running.
The other methods (AMReX PlotfileTools, yt, VisIt) allow you to analyze the outputs after they are written to disk.

Ascent
=====
Ascent allows you to generate visualizations (as PNG images) while the simulation is running, without any extra effort.

.. note:: On Setonix, Ascent is already built.
  In your job script, add the line:
  ``export Ascent_DIR=/software/projects/pawsey0807/bwibking/ascent_06082023/install/ascent-develop/lib/cmake/ascent``.

Compiling Ascent via Spack
-----------------------
1. Run ``spack external find``.
2. Make sure there are entries listed for ``hdf5``, ``cuda``, and ``openmpi`` in your ``~/.spack/packages.yaml`` file.
3. Add `buildable: False <https://spack.readthedocs.io/en/latest/build_settings.html#external-packages>`_ to each entry.
4. Run ``spack fetch --dependencies ascent@develop+cuda+vtkh~fortran~shared cuda_arch=70 ^conduit~parmetis~fortran``
5. On a dedicated compute node, run ``spack install ascent@develop+cuda+vtkh~fortran~shared cuda_arch=70 ^conduit~parmetis~fortran``

For A100 GPUs, change the above lines to `cuda_arch=80`.
Currently, it's not possible to `build for both GPU models at the same time <https://github.com/Alpine-DAV/ascent/issues/950#issuecomment-1153243232>`_.

Compiling Quokka with Ascent support
-----------------------
1. Load Ascent: ``spack load ascent``
2. Add ``-DAMReX_ASCENT=ON -DAMReX_CONDUIT=ON`` to your CMake options.
3. Compile your problem, e.g.: ``ninja -j4 test_hydro3d_blast``

Customizing the visualization
-----------------------
Add an `ascent_actions.yaml file <https://ascent.readthedocs.io/en/latest/Actions/Actions.html>`_ to the simulation working directory.
This file can even be edited while the simulation is running!

.. warning:: Volume renderings do not correctly handle ghost cells (`GitHub issue <https://github.com/Alpine-DAV/ascent/issues/955>`_).

AMReX PlotfileTools
=====
These are self-contained C++ programs (included with AMReX in the ``Tools/Plotfile`` subdirectory) that will output a 2D slice (axis-aligned), a 1D slice (axis-aligned), or compute a volume integral given an AMReX plotfile.
For these tasks, it is almost always easier to use the Plotfile tools rather than, e.g., yt or VisIt.

* To compute a volume integral, use `fvolumesum <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fvolumesum.cpp>`_.
* To compute a 2D slice plot (axis-aligned planes only), use `fsnapshot <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fsnapshot.cpp>`_.
* To compute a 1D slice (axis-aligned directions only, with output as ASCII), use `fextract <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fextract.cpp>`_.

Other tools:

* `fboxinfo <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fboxinfo.cpp>`_ prints out the indices of all the Boxes in a plotfile
* `fcompare <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fcompare.cpp>`_ calculates the absolute and relative errors between plotfiles in L-inf norm
* `fextrema <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fextrema.cpp>`_ calculates the minimum and maximum values of all variables in a plotfile
* `fnan <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fnan.cpp>`_ determines whether there are any NaNs in a plotfile
* `ftime <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/ftime.cpp>`_ prints the simulation time of each plotfile
* `fvarnames <https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fvarnames.cpp>`_ prints the names of all the variables in a given plotfile

yt
=====
.. warning:: There are `known bugs <https://github.com/yt-project/yt/issues/3889>`_ that affect Quokka outputs.
  PlotfileTools (see above) can be used instead for axis-aligned slice plots.

The plotfile directory can be loaded with ``yt.load`` as usual. However, the standard fields such as ``('gas', 'density')`` are not defined.
Instead, you have to use non-standard fields. Examine ``ds.field_list`` to see the fields that exist in the plotfiles. These should be: ::

  [('boxlib', 'gasDensity'), ('boxlib', 'gasEnergy'),
  ('boxlib', 'radEnergy'), ('boxlib', 'scalar'),
  ('boxlib', 'temperature'), ('boxlib', 'x-GasMomentum'),
  ('boxlib', 'x-RadFlux'), ('boxlib', 'y-GasMomentum'),
  ('boxlib', 'y-RadFlux'), ('boxlib', 'z-GasMomentum'), ('boxlib', 'z-RadFlux')]

For details, see the `yt documentation on reading AMReX data <https://yt-project.org/doc/examining/loading_data.html#amrex-boxlib-data>`_.

.. tip:: One of the most useful things to do is to convert the data into a uniform-resolution NumPy array
  with the `covering_grid <https://yt-project.org/doc/examining/low_level_inspection.html#examining-grid-data-in-a-fixed-resolution-array>`_ function.

.. tip:: This `WarpX script <https://warpx.readthedocs.io/en/latest/dataanalysis/plot_parallel.html>`_ may be useful as a starting point
  for visualizing a time series of outputs. This script will require some modification to work with Quokka outputs.

VisIt
=====
VisIt can read AMReX plotfiles. You have to select the ``plt00000/Header`` file in VisIt's Open dialog box.

.. warning:: There are some rendering bugs with unscaled box dimensions.
  Do not expect volume rendering to work when using, e.g. parsec-size boxes with cgs units.
