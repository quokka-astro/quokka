# shellcheck shell=bash disable=SC2034  # remote.sh reads these
# moth: one MI210 (gfx90a), ROCm 10.1, no batch system. remote.sh sources
# this as bin/site.sh; see README.md for what a site file defines.
SITE=moth
R=/mnt/data/bwibking
# Shared with the chemistry harness ($R/ar): one GPU job at a time on moth.
GPU_LOCK=$R/ar/gpu.lock
V=$R/venv-py312/bin
ROCM=/opt/rocm/core-10.1
ROCPROF=$ROCM/bin/rocprofv3
# Other users share moth's GPU: amd-smi finds their processes and samples clocks.
AMDSMI=$ROCM/bin/amd-smi
PYTHON=python3
# The MI210 sits on NUMA node 1 (CPUs 24-47). Builds stay on node 0; timing
# runs pin to two node-1 cores.
BUILD_PREFIX=(taskset -c 0-23)
BUILD_JOBS=24
GPU_PREFIX=(taskset -c 46,47)
CONFIGURE_ARGS=(-DCMAKE_MAKE_PROGRAM="$V/ninja"
  -DCMAKE_C_COMPILER="$ROCM/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM/llvm/bin/clang++"
  -DCMAKE_PREFIX_PATH="$ROCM")

site_env() {
  export XDG_CACHE_HOME=$R/cache TMPDIR=$R/tmp PYTHONPYCACHEPREFIX=$R/cache/pycache
  # Quokka forces AMReX_MPI=ON; moth's OpenMPI is not on the default PATH.
  export PIP_NO_CACHE_DIR=1 PATH=/usr/lib64/openmpi/bin:$V:$PATH
}
site_enter() { :; }  # ssh already lands on the GPU host
site_check() { :; }  # the GPU never goes away
