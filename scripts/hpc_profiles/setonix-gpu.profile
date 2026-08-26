#!/bin/bash

source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh

module load cpe/26.03
module load pawseyenv/2025.08

module load PrgEnv-cray
module load craype-x86-trento
module load craype-accel-amd-gfx90a

module load cce/21.0.2
module load cray-mpich/9.1.0

# singularity
module load singularity/4.1.0-slurm

# hdf5
module load cray-hdf5

# python
module load cray-python/3.12.12

# ROCm 7.14 user-space SDK for the MI250X (gfx90a). The host amdgpu
# driver is supplied by Setonix; install the wheel environment only once.
ROCM_VENV="${MYSOFTWARE}/venvs/rocm-7.14"

# Keep pip's large downloads and temporary build files out of $HOME, where
# Setonix applies a comparatively small quota. Pawsey defines $MYSCRATCH and
# $MYSOFTWARE for each user/project, so no account-specific paths are needed.
export PIP_CACHE_DIR="${MYSCRATCH:?MYSCRATCH is not set}/pip-cache"
export TMPDIR="${MYSCRATCH}/pip-tmp"
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

# cmake
if ! python -c 'import cmake' 2>/dev/null; then
  python -m pip install cmake
fi

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# optimize ROCm/HIP compilation for MI250X
export AMREX_AMD_ARCH=gfx90a

# compiler environment hints
export CC=$(which hipcc)
export CXX=$(which hipcc)
export FC=$(which ftn)

# these flags are REQUIRED
export CFLAGS="-I${MPICH_DIR}/include --gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14"
export CXXFLAGS="-I${MPICH_DIR}/include --gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14"
export LDFLAGS="-L${MPICH_DIR}/lib -lmpi \
  ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem \
  ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"
