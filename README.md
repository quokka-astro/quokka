[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=alert_status&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=bugs&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=BenWibking_TwoMomentRad&metric=ncloc&token=5049c56ffe08dcc83afd5ca4c8e0d951a2836652)](https://sonarcloud.io/dashboard?id=BenWibking_TwoMomentRad)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)

# QUOKKA
*Quadrilateral, Umbra-producing, Orthogonal, Kangaroo-conserving Kode for Astrophysics!*

**The Quokka methods paper is now available: https://arxiv.org/abs/2110.01792**

**Please see the sections "Quickstart" and "Running on GPUs" below. You can start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support, or open an [Issue](https://github.com/BenWibking/quokka/issues) for any bug reports.**

Quokka is a two-moment radiation hydrodynamics code that uses the piecewise-parabolic method, with AMR and subcycling in time. Runs on CPUs (MPI+vectorized) or NVIDIA GPUs (MPI+CUDA) with a single-source codebase. Written in C++17. (100% Fortran-free.)

Here is a [a Kelvin-Helmholz instability simulated with Quokka](https://vimeo.com/714653592) on a 512x512 uniform grid:

![Animated GIF of KH Instability](https://videoapi-muybridge.vimeocdn.com/animated-thumbnails/image/1f468be6-6d7b-4d53-a02c-4dd8f3ad5154.gif?ClientID=vimeo-core-prod&Date=1653705774&Signature=9bea89d5c9657180391a9538a10fd4f8f7099025)

This is [a 3D Rayleigh-Taylor instability](https://vimeo.com/746363534) simulated on a $256^3$ grid:

![Image of 3D RT instability](extern/rt3d_visit.png)

Quokka also features advanced Adaptive Quokka Refinement:tm: technology:

![Image of Quokka with Baby in Pouch](extern/quokka2.png)

## Dependencies
* C++ compiler (with C++17 support)
* CMake 3.16+
* MPI library with GPU-aware support (OpenMPI, MPICH, or Cray MPI)
* HDF5 1.10+ (serial version)
* CUDA 11.7+ (optional, for NVIDIA GPUs)
* ROCm 5.2.0+ (optional, for AMD GPUs)
* Ninja (optional, for faster builds)
* Python 3.7+ (optional)

## Quickstart

To run Quokka, download this repository to your local machine:
```
git clone --recursive https://github.com/BenWibking/quokka.git
```
Quokka uses CMake (and optionally, Ninja) as its build system. If you don't have CMake and Ninja installed, the easiest way to install them is to run:
```
python3 -m pip install cmake ninja --user
```
Now that CMake is installed, create a `build/` subdirectory and compile Quokka, as shown below.
```
cd quokka
mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -j6
```
Congratuations! You have now built all of the 1D test problems on CPU. You can run the automated test suite:
```
ninja test
```
You should see output that indicates all tests have passed, like this:
```
100% tests passed, 0 tests failed out of 20

Total Test time (real) = 111.74 sec
```
To run in 2D or 3D, build with the `-DAMReX_SPACEDIM` CMake option, for example:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3 -G Ninja
ninja -j6
```
to compile Quokka for 3D problems.

**By default, Quokka compiles itself *only* for CPUs. If you want to run Quokka on GPUs, see the section "Running on GPUs" below.**

Have fun!

## Building with CMake + `make`
If you are unable to install Ninja, you can instead use CMake with the Makefile generator, which should produce identical results but is slower:
```
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
make -j6
make test
```

## Could NOT find Python error
If CMake prints an error saying that Python could not be found, e.g.:
```
-- Could NOT find Python (missing: Python_EXECUTABLE Python_INCLUDE_DIRS Python_LIBRARIES Python_NumPy_INCLUDE_DIRS Interpreter Development NumPy Development.Module Development.Embed)
```
you should be able to fix this by installing NumPy (and matplotlib) by running
```
python3 -m pip install numpy matplotlib --user
```
This should enable CMake to find the NumPy header files that are needed to successfully compile.

Alternatively, you can work around this problem by disabling Python support. Python and NumPy are only used to plot the results of some test problems, so this does not otherwise affect Quokka's functionality. Add the option
```
-DQUOKKA_PYTHON=OFF
```
to the CMake command-line options (or change the `QUOKKA_PYTHON` option to `OFF` in CMakeLists.txt).

## Running on GPUs
By default, Quokka compiles itself to run only on CPUs. Quokka can run on either NVIDIA, AMD, or Intel GPUs. Consult the sub-sections below for the build instructions for a given GPU vendor.

### NVIDIA GPUs
If you want to run on NVIDIA GPUs, re-build Quokka as shown below. (*CUDA >= 11.7 is required. Quokka is only supported on Volta V100 GPUs or newer models. Your MPI library **must** support CUDA-aware MPI.*)
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA -DAMREX_GPUS_PER_NODE=N -G Ninja
ninja -j6
```
where $N$ is the number of GPUs available per compute node.

**It is necessary to use `-DAMREX_GPUS_PER_NODE` to specify the number of GPUs per compute node. Without this, performance will be very poor. All GPUs on a node must be visible from each MPI rank on the node for efficient GPU-aware MPI communication to take place via CUDA IPC.** When using the SLURM job scheduler, this means that `--gpu-bind` should be set to `none`.

The compiled test problems are in the test problem subdirectories in `build/src/`. Example scripts for running Quokka on compute clusters are in the `scripts/` subdirectory.

Note that 1D problems can run very slowly on GPUs due to a lack of sufficient parallelism. To run the test suite in a reasonable amount of time, you may wish to exclude the matter-energy exchange tests, e.g.:
```
ctest -E "MatterEnergyExchange*"
```
which should end with output similar to the following:
```
100% tests passed, 0 tests failed out of 18

Total Test time (real) = 353.77 sec
```
### AMD GPUs
Compile with `-DAMReX_GPU_BACKEND=HIP`. Requires ROCm 5.2.0 or newer. Your MPI library **must** support GPU-aware MPI for AMD GPUs. Quokka has been tested on MI100 and MI250X GPUs.

### Intel GPUs
Not tested. You can attempt this by compiling with `-DAMReX_GPU_BACKEND=SYCL`. Please start a Discussion if you encounter issues on Intel GPUs. Your MPI library **must** support GPU-aware MPI for Intel GPUs.

## Building a specific test problem
By default, all available test problems will be compiled. If you only want to build a specific problem, you can list all of the available CMake targets:
```
cmake --build . --target help
```
and then build the problem of interest:
```
ninja -j6 test_hydro3d_blast
```

## Problems?
If you run into problems, please start a [Discussion](https://github.com/BenWibking/quokka/discussions) for technical support. If you discover a bug, please let us know by opening an [Issue](https://github.com/BenWibking/quokka/issues).
