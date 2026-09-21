#!/bin/bash
#SBATCH --job-name=ctest
#SBATCH --cpus-per-task=1
#SBATCH --time=12:30:00
#SBATCH -o /data/mfulghieri/ufficial_quokka/outputs/compilation/ctest.out
#SBATCH -e /data/mfulghieri/ufficial_quokka/outputs/compilation/ctest.err

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

# Network setting for OpenMPI in local/shared memory execution (eliminates OpenFabrics network warnings)
export OMPI_MCA_btl="vader,self"  # MPI processes communicate only via local shared memory (vader) or self
export OMPI_MCA_rmaps_base_oversubscribe=1  # Enable MPI to oversubscribe if a test requires more slots

echo "=== SLURM DIAGNOSTIC ==="
echo "CPUs per task required: $SLURM_CPUS_PER_TASK"
echo "Total tasks:            $SLURM_NTASKS"
echo "Assigned nodes:         $SLURM_JOB_NODELIST"
env | grep -E "SLURM|OMPI|MPI"


# Run tests via ctest with allocated Slurm threads (default 30 if run outside Slurm)
# Additional arguments can be passed via command line thanks to "$@" (e.g. ./ctest.sh -R Hydro)
ctest -j"${SLURM_CPUS_PER_TASK:-30}" --output-on-failure "$@"


