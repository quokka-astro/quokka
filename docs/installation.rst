.. Installation

Installation
============

To run Quokka, download this repository to your local machine::

    git clone git@github.com:BenWibking/quokka.git

Then download all submodules (this downloads `AMReX` and the string-formatting library `fmt`)::

    cd quokka
    git submodule update --init

Create a build/ subdirectory and compile Quokka::

    mkdir build; cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j6

Congratuations! You have now built all of the 1D test problems on CPU. You can run the automated test suite::

    make test -j6

You should see output that indicates all tests have passed, like this::

    100% tests passed, 0 tests failed out of 20

    Total Test time (real) = 111.74 sec

To run in 2D or 3D, edit the `AMReX_SPACEDIM` option in the `CMakeLists.txt` file, for example::

    set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)

to compile Quokka for 3D problems.

Have fun!

Running on GPUs
---------------

By default, Quokka compiles itself to run only on CPUs. If you want to run on NVIDIA GPUs, re-build Quokka with the following options::

    cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA
    make -j6

The compiled test problems are in the test problem subdirectories in `build/src/`. Example scripts for running Quokka on compute clusters are in the `scripts/` subdirectory. Please note that you must configure your compute cluster to run with 1 MPI rank per GPU in order for Quokka to work correctly. Quokka is only supported on Volta-class (V100) GPUs or newer.

**AMD or Intel GPUs:** Running on AMD or Intel GPUs is currently experimental and has *not been tested* by the Quokka developers. AMReX is currently undergoing rapid advances in its support for GPUs from these vendors, so please get in touch by starting a Discussion before attempting this.
