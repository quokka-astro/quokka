# Postprocessing

There are several ways to post-process the output of Quokka simulations. AMReX PlotfileTools, yt, and VisIt all allow you to analyze the outputs after they are written to disk.

## Amrvis-container

[Amrvis-container](https://github.com/AMReX-Codes/Amrvis-container) bundles Amrvis in a Docker/Apptainer image with a browser-based X11 frontend. To browse Quokka plotfiles locally (Docker required), run from the Amrvis-container repo:

    ./launch_amrvis_browser.sh /path/to/plotfiles

The target directory is bind-mounted to `/home/vscode/data` in the container. The launcher prints a one-time password; open `http://localhost:8080`, paste the password, and use the `xterm` window to start `amrvis2d` or `amrvis3d` on your `plt*` directories.

!!! Tip
    On SLURM clusters with Apptainer, pull the image once with `apptainer pull amrvis-container.sif docker://ghcr.io/amrex-codes/amrvis-container:main`, then use `./launch_amrvis_browser_hpc.sh /path/to/plotfiles` on a compute node and follow the printed SSH tunnel instructions.

## AMReX PlotfileTools

These are self-contained C++ programs (included with AMReX in the `Tools/Plotfile` subdirectory) that will output a 2D slice (axis-aligned), a 1D slice (axis-aligned), or compute a volume integral given an AMReX plotfile. This works as an alternative to yt and VisIt for basic tasks.

-   To compute a volume integral, use [fvolumesum](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fvolumesum.cpp).
-   To compute a 2D slice plot (axis-aligned planes only), use [fsnapshot](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fsnapshot.cpp).
-   To compute a 1D slice (axis-aligned directions only, with output as ASCII), use [fextract](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fextract.cpp).

Other tools:

-   [fboxinfo](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fboxinfo.cpp) prints out the indices of all the Boxes in a plotfile
-   [fcompare](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fcompare.cpp) calculates the absolute and relative errors between plotfiles in L-inf norm
-   [fextrema](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fextrema.cpp) calculates the minimum and maximum values of all variables in a plotfile
-   [fnan](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fnan.cpp) determines whether there are any NaNs in a plotfile
-   [ftime](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/ftime.cpp) prints the simulation time of each plotfile
-   [fvarnames](https://github.com/AMReX-Codes/amrex/blob/development/Tools/Plotfile/fvarnames.cpp) prints the names of all the variables in a given plotfile

## yt

!!! Warning
    There are [known bugs](https://github.com/yt-project/yt/issues/3889) that affect Quokka outputs. PlotfileTools (see above) can be used instead for axis-aligned slice plots.

!!! Tip
    One of the most useful things to do is to convert the data into a uniform-resolution NumPy array with the [covering_grid](https://yt-project.org/doc/examining/low_level_inspection.html#examining-grid-data-in-a-fixed-resolution-array) function.

We have a fork of YT that includes a customized Quokka frontend: [https://github.com/chongchonghe/yt](https://github.com/chongchonghe/yt). To install it, run `pip install "yt[quokka] @ git+https://github.com/chongchonghe/yt.git"`. A comprehensive documentation is available at [this link](https://github.com/chongchonghe/yt/blob/Rongjun-ANUquokka-frontend/doc/source/examining/loading_data.rst#quokka-data), and a Jupyter Notebook with tutorials is available at [README.ipynb](https://github.com/Rongjun-ANU/README-of-yt-frontend-for-QUOKKA/blob/main/README.ipynb).

The `quick_plot` script in `scripts/python/` is a convenient tool for visualizing Quokka outputs. It is a wrapper around YT for batch processing snapshots and generating slice or projection plots. The script has detailed documentation in the code itself, accessible at the top of the file and also by running `quick_plot -h`.

## VisIt

VisIt can read cell-centered output variables from AMReX plotfiles. Currently, there is no support for reading either face-centered variables or particles. (However, by default, cell-centered averages of face-centered variables are included in Quokka plotfiles.)

In order to read an individual plotfile, you can select the `plt00000/Header` file in VisIt's Open dialog box.

If you want to read a timeseries of plotfiles, you can create a file with a ``.visit`` extension that lists the ``plt*/Header`` files, one per line, with the following command: :

    ls -1 plt*/Header | tee plotfiles.visit

Then select ``plotfiles.visit`` in VisIt's Open dialog box.

!!! Warning
    There are rendering bugs with unscaled box dimensions. Slices generally work. However, do not expect volume rendering to work when using, e.g. parsec-size boxes with cgs units.
