#!/bin/bash
#SBATCH --job-name=ninja
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:30:00
#SBATCH -o /data/mfulghieri/ufficial_quokka/outputs/compilation/CMake_NinjaCompile.out
#SBATCH -e /data/mfulghieri/ufficial_quokka/outputs/compilation/CMake_NinjaCompile.err

# Determine repository root directory safely (avoiding Slurm /var/spool issue)
REPO_DIR=""
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_DIR="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null)"
fi
if [ -z "$REPO_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
fi
if [ -z "$REPO_DIR" ] || [[ "$REPO_DIR" == "/var/spool"* ]]; then
    REPO_DIR="/data/mfulghieri/ufficial_quokka"
fi
BUILD_DIR="$REPO_DIR/build"

# Ensure output directory for logs exists
mkdir -p "$REPO_DIR/outputs/compilation"

# Ensure build directory exists and navigate to it
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# Clean Conda from environment if active to avoid ABI/library conflicts
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | grep -v anaconda | tr '\n' ':' | sed 's/:$//')
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE
fi

# Load cluster HPC modules (MPI, HDF5, CMake), then override compiler with GCC 14.2.0 & Ninja
module purge
module load gcc-11.3.0/ompi-4.1.4_nccl
module load gcc-11.3.0/hdf5-1.14.1
module load cmake-3.22.1
module use /data/mfulghieri/mymodules
module load mygcc/14.2.0
module load ninja/1.12.1

# Ensure OpenMPI wrappers explicitly use GCC 14.2.0
export OMPI_CC="$(which gcc)"
export OMPI_CXX="$(which g++)"

printf "\n============= PATH INSPECTION =============\n"
echo "PATH content: $PATH"
echo "'which gcc':    $(which gcc)"
echo "'which g++':    $(which g++)"
echo "'which ninja':  $(which ninja)"
echo "'which mpicxx': $(which mpicxx)"
printf "============================================\n\n"

# Configure with CMake using Ninja generator and GCC 14.2.0 compiler (isolate from Conda headers/libs)
cmake "$REPO_DIR" \
  -DCMAKE_C_COMPILER="$(which gcc)" \
  -DCMAKE_CXX_COMPILER="$(which g++)" \
  -DQUOKKA_PYTHON=OFF \
  -DCMAKE_IGNORE_PATH="/data/mfulghieri/anaconda3/envs/quokka/lib;/data/mfulghieri/anaconda3/envs/quokka/include" \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMReX_SPACEDIM=3 \
  -G Ninja 

# Build with Ninja using allocated Slurm cpus (default 30 if run outside Slurm)
ninja -j"${SLURM_CPUS_PER_TASK:-30}"