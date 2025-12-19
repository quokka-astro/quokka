# Installation on macOS

This guide provides detailed instructions for installing and building Quokka on macOS systems.

## Prerequisites

Before installing Quokka, you need to ensure that you have a working C++ compiler, MPI library, CMake, and Ninja installed on your system.

### Step 1: Verify C++ Compiler

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

### Step 2: Verify and Install MPI

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

### Step 3: Install CMake and Ninja

You can install CMake and Ninja using either pip or Homebrew.

**Option 1: Install via pip or uv**

pip install:
```bash
python3 -m pip install cmake ninja --user
```

uv install:
```bash
uv add cmake ninja --dev
```

**Option 2: Install via Homebrew**

```bash
brew install cmake ninja
```

Verify the installation:

```bash
cmake --version
ninja --version
```

### Step 4: Install Python Dependencies (Optional but Recommended)

Some test problems use Python for plotting results. Install NumPy and matplotlib:

```bash
python3 -m pip install numpy matplotlib --user
```

If you skip this step, you can disable Python support later by adding `-DQUOKKA_PYTHON=OFF` to the CMake configuration.

## Building Quokka

Following the instructions in [Installation](installation.md).

## Troubleshooting

### MPI Compiler Issues

If you encounter issues with the MPI compiler, you can explicitly specify it:

```bash
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc -DCMAKE_BUILD_TYPE=Release -G Ninja
```
