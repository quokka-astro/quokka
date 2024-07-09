.. Postprocessing

Postprocessing
=====

There are several ways to post-process the output of Quokka simulations.
AMReX PlotfileTools, yt, and VisIt all allow you to analyze the outputs after they are written to disk.

AMReX PlotfileTools
-----------------------
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
-----------------------
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
-----------------------
VisIt can read cell-centered output variables from AMReX plotfiles. Currently, there is no support for reading either face-centered variables or particles. (However, by default, cell-centered averages of face-centered variables are included in Quokka plotfiles.)

In order to read an individual plotfile, you can select the ``plt00000/Header`` file in VisIt's Open dialog box.

If you want to read a timeseries of plotfiles, you can create a file with a `.visit` extension that lists the `plt*/Header` files, one per line, with the following command: ::

  ls -1 plt*/Header | tee plotfiles.visit

Then select `plotfiles.visit` in VisIt's Open dialog box.

.. warning:: There are rendering bugs with unscaled box dimensions. Slices generally work.
  However, do not expect volume rendering to work when using, e.g. parsec-size boxes with cgs units.
