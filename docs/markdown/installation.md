# Installation

## Building on Linux and macOS

This instruction works on both Linux and macOS. If you come across any issues on macOS, see [Installation on macOS](#installation-on-macos) for detailed instructions.

To run Quokka, download this repository and its submodules to your local machine:

    git clone --recursive https://github.com/quokka-astro/quokka.git

Quokka uses CMake (and optionally, Ninja) as its build system. If you don't have CMake and Ninja installed, the easiest way to install them is to run:

    python3 -m pip install cmake ninja --user

Alternatively, if you have [uv](https://docs.astral.sh/uv/) installed, you can use:

    uv pip install cmake ninja

Now that CMake is installed, create a ``build/`` subdirectory and compile Quokka, as shown below.

    cd quokka
    mkdir build; cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
    ninja -j6

Congratuations! You have now built all of the 1D test problems on CPU. You can run the automated test suite:

    ninja test

You should see output that indicates all tests have passed, like this:

    100% tests passed, 0 tests failed out of 20

    Total Test time (real) = 111.74 sec

To run in 2D or 3D, build with the ``-DAMReX_SPACEDIM`` CMake option, for example:

    cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3 -G Ninja
    ninja -j6

to compile Quokka for 3D problems.

**By default, Quokka compiles itself only for CPUs. If you want to run Quokka on GPUs, see the section "Running on GPUs" below.**

Have fun!

## Building with CMake + `make`

If you are unable to install Ninja, you can instead use CMake with the Makefile generator, which should produce identical results but is slower:

    cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
    make -j6
    make test

## Could NOT find Python error

If CMake prints an error saying that Python could not be found, e.g.:

    -- Could NOT find Python (missing: Python_EXECUTABLE Python_INCLUDE_DIRS Python_LIBRARIES Python_NumPy_INCLUDE_DIRS Interpreter Development NumPy Development.Module Development.Embed)

you should be able to fix this by installing NumPy (and matplotlib) by running

    python3 -m pip install numpy matplotlib --user

or with uv:

    uv pip install numpy matplotlib

This should enable CMake to find the NumPy header files that are needed to successfully compile.

Alternatively, you can work around this problem by disabling Python support. Python and NumPy are only used to plot the results of some test problems, so this does not otherwise affect Quokka's functionality. Add the option

    -DQUOKKA_PYTHON=OFF

to the CMake command-line options (or change the `QUOKKA_PYTHON` option to `OFF` in CMakeLists.txt).

## Running on GPUs

By default, Quokka compiles itself to run only on CPUs. Quokka can run on either NVIDIA or AMD GPUs. Consult the sub-sections below for the build instructions for a given GPU vendor.

### NVIDIA GPUs

If you want to run on NVIDIA GPUs, re-build Quokka as shown below. (*CUDA >= 11.7 is required. Quokka is only supported on Volta V100 GPUs or newer models. Your MPI library* **must** *support CUDA-aware MPI.*)

    cmake .. -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA -DAMReX_SPACEDIM=3 -G Ninja
    ninja -j6

**All GPUs on a node must be visible from each MPI rank on the node for efficient GPU-aware MPI communication to take place via CUDA IPC.** When using the SLURM job scheduler, this means that `--gpu-bind` should be set to `none`.

The compiled test problems are in the test problem subdirectories in `build/src/`. Example scripts for running Quokka on compute clusters are in the `scripts/` subdirectory.

Note that 1D problems can run very slowly on GPUs due to a lack of sufficient parallelism. To run the test suite in a reasonable amount of time, you may wish to exclude the matter-energy exchange tests, e.g.:

    ctest -E "MatterEnergyExchange*"

which should end with output similar to the following:

    100% tests passed, 0 tests failed out of 18

    Total Test time (real) = 353.77 sec

### AMD GPUs

> *Requires ROCm 6.3.0 or newer. The directory containing the HIP and other related binaries must be added to the `PATH` environment variable after the ROCm installation.*

Build with `-DAMReX_GPU_BACKEND=HIP`. Your MPI library **must** support GPU-aware MPI for AMD GPUs. The typical AMD GPU compilers are `amdclang++` or `hipcc`. In case your GPU-aware compiler is not being used by default during the build, use the `DCMAKE_CXX_COMPILER` and `DCMAKE_C_COMPILER` options to specify the C++ and C compilers respectively. Additionally, the AMD GPU architecture may have to be specified. This can be done using the `DAMReX_GPU_ARCH` option. The GPU architecture can be found using

```shell
rocminfo | grep gfx
```

A typical build command using the `amdclang++` compiler and an AMD GPU with RDNA 2 (gfx1031) architecture will look like

```shell
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=amdclang++ \
         -DCMAKE_C_COMPILER=amdclang \
         -DAMReX_GPU_BACKEND=HIP \
         -DAMReX_GPU_ARCH=gfx1031 \
         -G Ninja
```

Quokka has been tested on MI100, MI250X and 6700XT GPUs.

### Intel GPUs *(does not compile)*

Due to limitations in the Intel GPU programming model, Quokka currently cannot be compiled for Intel GPUs. (See <https://github.com/quokka-astro/quokka/issues/619> for the technical details.)

## Building a specific test problem

By default, all available test problems will be compiled. If you only want to build a specific problem, you can list all of the available CMake targets:

    cmake --build . --target help

and then build the problem of interest:

    ninja -j6 test_hydro3d_blast


## Installation on macOS

This guide provides detailed instructions for installing and building Quokka on macOS systems.

### Prerequisites

Before installing Quokka, you need to ensure that you have a working C++ compiler, MPI library, CMake, and Ninja installed on your system.

#### Step 1: Verify C++ Compiler

First, check if you have a working C++ compiler installed by compiling a simple program:

```bash
cat > /tmp/cpp.cpp <<'EOF'
#include <iostream>
int main(){ std::cout << "C++ works\n"; }
EOF
clang++ /tmp/cpp.cpp -o /tmp/cpp && /tmp/cpp
```

If this command succeeds and prints "C++ works", you're good to go. If not, you'll need to install Xcode Command Line Tools:

```bash
xcode-select --install
```

Follow the prompts to complete the installation, then verify C++ works again using the test above.

#### Step 2: Verify and Install MPI

Check if MPI is already installed:

```bash
mpicxx --version
```

If MPI is not installed, install it using Homebrew:

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Open MPI
brew install open-mpi
```

After installation, verify that MPI works correctly:

```bash
mpicxx --version
mpicxx --show

cat > /tmp/mpi_cpp.cpp <<'EOF'
#include <mpi.h>
#include <iostream>
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  int r; MPI_Comm_rank(MPI_COMM_WORLD,&r);
  std::cout<<"Hello from C++ rank "<<r<<"\n";
  MPI_Finalize();
  return 0;
}
EOF

mpicxx /tmp/mpi_cpp.cpp -o /tmp/mpi_cpp && /tmp/mpi_cpp
```

This should compile and run successfully, printing "Hello from C++ rank 0".

#### Step 3: Install CMake and Ninja

You can install CMake and Ninja using either pip or Homebrew.

**Option 1: Install via pip or uv**

Using pip:
```bash
python3 -m pip install cmake ninja --user
```

Using uv (if you have [uv](https://docs.astral.sh/uv/) installed):
```bash
uv tool install cmake ninja
```

Note: For uv, you may also use `uv pip install cmake ninja` if you're working within a virtual environment.

**Option 2: Install via Homebrew**

```bash
brew install cmake ninja
```

Verify the installation:

```bash
cmake --version
ninja --version
```

#### Step 4: Install Python Dependencies (Optional but Recommended)

Some test problems use Python for plotting results. Install NumPy and matplotlib:

Using pip:
```bash
python3 -m pip install numpy matplotlib --user
```

Using uv:
```bash
uv pip install numpy matplotlib
```

If you skip this step, you can disable Python support later by adding `-DQUOKKA_PYTHON=OFF` to the CMake configuration.

### Building Quokka

Following the instructions in [Installation](installation.md).

### Troubleshooting

#### MPI Compiler Issues

If you encounter issues with the MPI compiler, you can explicitly specify it:

```bash
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_BUILD_TYPE=Release -G Ninja
```
