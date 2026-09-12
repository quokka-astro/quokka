#!/bin/bash
# please set your project account
export proj="ast236"  # change me!

## modules
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh
module load Core/26.05
module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load cce/21.0.0
module load cray-mpich/9.1.0
module load cray-hdf5/1.14.3.9
module load cray-python/3.12.12
module load cmake/4.1.5
# adios2 (optional)
module load adios2/2.11.0-mpi
# emacs (optional)
module load emacs

## aliases
alias getNode="salloc -A $proj -J quokka -t 01:00:00 -p batch -N 1"
#   usage: runNode <command>
alias runNode="srun -A $proj -J quokka -t 00:30:00 -p batch -N 1"
alias snodes="sinfo -O PartitionName:12,StateComplete:50,Nodes:10,Reason:90 -S +P,+E,+t"
alias savail="sinfo -O PartitionName:12,Nodes:10,StateComplete:50 -S +P,+E,+t -t alloc,idle"

## ROCm 7.14.0
# ROCm 7.14 user-space SDK for the MI250X (gfx90a). The host amdgpu driver is supplied by Frontier
ROCM_VENV="${HOME}/venvs/rocm-7.14"

# Keep pip's large downloads and temporary build files out of $HOME, where
# Setonix applies a comparatively small quota. Pawsey defines $MYSCRATCH
# for each user/project, so no account-specific paths are needed.
export PIP_CACHE_DIR="/tmp/${USER}/pip-cache"
export TMPDIR="/tmp/${USER}/pip-tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" "$(dirname "${ROCM_VENV}")"

# Checking only bin/python is insufficient: a failed pip invocation leaves a
# valid but incomplete virtual environment. Retry until the target SDK exists.
if [[ ! -x "${ROCM_VENV}/bin/python" ]] ||
   ! "${ROCM_VENV}/bin/python" -m pip show rocm-sdk-device-gfx90a \
     >/dev/null 2>&1; then
  python -m venv "${ROCM_VENV}"
  "${ROCM_VENV}/bin/python" -m pip install --upgrade pip
  "${ROCM_VENV}/bin/python" -m pip install \
    --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    "rocm[libraries,devel,device-gfx90a]==7.14.0"
  fi
source "${ROCM_VENV}/bin/activate"

echo "Initializing ROCm SDK... (may take several minutes on a parallel file system, please be patient)"
rocm-sdk init

# Expose the wheel-provided ROCm installation to CMake and AMReX. The SDK
# activation places hipcc on PATH but does not currently publish its CMake
# package prefix or HIP installation path.
ROCM_SDK_ROOT="${ROCM_VENV}/lib/python3.12/site-packages/_rocm_sdk_devel"
export ROCM_PATH="${ROCM_SDK_ROOT}"
export HIP_PATH="${ROCM_SDK_ROOT}"
export CMAKE_PREFIX_PATH="${ROCM_SDK_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${ROCM_SDK_ROOT}/lib:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

## environment variables

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a
# compilers
export CC=$(which hipcc)
export CXX=$(which hipcc)

# These flags are required when using hipcc directly instead of the Cray
# compiler wrappers. In particular, link the GPU Transport Layer (GTL) needed
# by GPU-aware MPICH on gfx90a.
export CFLAGS="-I${MPICH_DIR}/include"
export CXXFLAGS="-I${MPICH_DIR}/include"
export LDFLAGS="-L${MPICH_DIR}/lib -lmpi \
  ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem \
  ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
