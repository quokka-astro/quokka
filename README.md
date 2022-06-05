[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=alert_status&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=bugs&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=ncloc&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Documentation Status](https://readthedocs.org/projects/quokka-code/badge/?version=latest)](https://quokka-code.readthedocs.io/en/latest/?badge=latest)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)

# Quokka
*Quadrilateral, Umbra-producing, Orthogonal, Kangaroo-conserving Kode for Astrophysics!*

**The Quokka methods paper is now available: https://arxiv.org/abs/2110.01792**

**NOTE: The code documentation is still a work in progress. Please see the Installation Notes below. You can start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support, or open an [Issue](https://github.com/BenWibking/quokka/issues) for any bug reports.**

Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++17. (100% Fortran-free.)

A [512x512 simulation of a Kelvin-Helmholz instability](https://vimeo.com/714653592):
![Animated GIF of KH Instability](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/1f468be6-6d7b-4d53-a02c-4dd8f3ad5154.gif?ClientID=vimeo-core-prod&Date=1653705774&Signature=9bea89d5c9657180391a9538a10fd4f8f7099025)

...with advanced Adaptive Quokka Refinement:tm: technology:

![Image of Quokka with Baby in Pouch](extern/quokka2.png)

## Installation Notes

To run Quokka, download this repository to your local machine:
```
git clone --recursive git@github.com:BenWibking/quokka.git
```
Quokka uses CMake as its build system. If you don't have CMake installed, the easiest way to install it is to run:
```
pip install cmake --user
```
Now that CMake is installed, create a build/ subdirectory and compile Quokka, as shown below. **(Warning to Intel compiler users: the 'classic' Intel compilers `icc` and `icpc` generate incorrect code; see issue [5](https://github.com/BenWibking/quokka/issues/5). Use the newer `icpx` Intel compiler instead by [setting the CMAKE_C_COMPILER and CMAKE_CXX_COMPILER options](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html).)**
```
cd quokka
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j6
```
Congratuations! You have now built all of the 1D test problems on CPU. You can run the automated test suite:
```
make test
```
You should see output that indicates all tests have passed, like this:
```
100% tests passed, 0 tests failed out of 20

Total Test time (real) = 111.74 sec
```
To run in 2D or 3D, edit the `AMReX_SPACEDIM` option in the `CMakeLists.txt` file, for example:
```
set(AMReX_SPACEDIM 3 CACHE STRING "" FORCE)
```
to compile Quokka for 3D problems.

Have fun!

## CMake
Quokka uses CMake for its build system. If you don't have CMake installed already, you can simply `pip install cmake` or use another [installation method](https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html). If you are unfamiliar with CMake, [this tutorial](https://hsf-training.github.io/hsf-training-cmake-webpage/) may be useful.

## Python header file not found error
If you get an error saying that a NumPy header file is not found, you can work around this problem by disabling Python support. Python and NumPy are only used to plot the results of some test problems, so this does not otherwise affect Quokka's functionality. Add the option
```
-DQUOKKA_PYTHON=OFF
```
to the CMake command-line options (or change the `QUOKKA_PYTHON` option to `OFF` in CMakeLists.txt).

## Running on GPUs
By default, Quokka compiles itself to run only on CPUs. (If you want to run on NVIDIA GPUs, re-build Quokka as shown below. **(Warning: CUDA 11.6 generates invalid device code; see issue [21](https://github.com/BenWibking/quokka/issues/21). Use CUDA <= 11.5 instead.)**
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA
make -j6
```
The compiled test problems are in the test problem subdirectories in `build/src/`. Example scripts for running Quokka on compute clusters are in the `scripts/` subdirectory. Please note that you must configure your compute cluster to run with 1 MPI rank per GPU in order for Quokka to work correctly. Quokka is only supported on Volta-class (V100) GPUs or newer.

Note that 1D problems can run very slowly on GPUs due to a lack of sufficient parallelism. To run the test suite in a reasonable amount of time, you may wish to exclude the matter-energy exchange tests, e.g.:
```
ctest -E "MatterEnergyExchange*"
```
which should end with output similar to the following:
```
100% tests passed, 0 tests failed out of 18

Total Test time (real) = 353.77 sec
```

**AMD or Intel GPUs:** Running on AMD or Intel GPUs is currently experimental and has *not been tested* by the Quokka developers. AMReX is currently undergoing rapid advances in its support for GPUs from these vendors, so please get in touch by starting a Discussion before attempting this.

## Problems?
If you run into problems, please start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support. If you discover a bug, please let us know by opening an [Issue](https://github.com/BenWibking/quokka/issues).
