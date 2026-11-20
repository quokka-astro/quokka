# shellcheck shell=bash disable=SC2034  # remote.sh reads these
# Setonix (Pawsey): one MI250X GCD (gfx90a) in your interactive gpu-dev job.
# remote.sh sources this as bin/site.sh; see README.md for what a site file
# defines. ar's ssh lands on a login node, and site_enter re-runs remote.sh on
# the job's node, so builds, runs, the detached job and the GPU lock all live
# there. ROCm 10.1 comes from AMD's pip wheels in $VENV (README.md has the
# install); the system modules give cray-mpich and HDF5.
SITE=setonix
VENV=$MYSOFTWARE/ar-quokka/venv-rocm-10.1
# Node-local: only campaigns sharing one job contend for its GPU.
GPU_LOCK=/tmp/ar-quokka-$USER/gpu.lock
PYTHON="$VENV/bin/python -s"
# The job's cgroup already confines us to its cores and GCD.
BUILD_PREFIX=()
GPU_PREFIX=()
PARTITION=gpu-dev

site_enter() {  # site_enter ARGS... : re-run `remote.sh ARGS...` on the gpu-dev job's node
  if [[ $(hostname) == nid* ]]; then
    export AR_JOBID=${AR_JOBID:-${SLURM_JOB_ID:-$(squeue -h -u "$USER" -t RUNNING -w "$(hostname)" -o %i | head -1)}}
    return 0
  fi
  local job=${AR_JOBID:-} node
  if [[ -z $job ]]; then
    job=$(squeue -h -u "$USER" -t RUNNING -p "$PARTITION" -o %i)
    case $(wc -w <<<"$job") in
      0) echo "no running $PARTITION job: start one, e.g. salloc -p $PARTITION -A ${PAWSEY_PROJECT}-gpu --gpus-per-node=1 -t 4:00:00" >&2; return 1 ;;
      1) ;;
      *) echo "several running $PARTITION jobs (${job//$'\n'/ }): set AR_JOBID" >&2; return 1 ;;
    esac
  fi
  node=$(squeue -h -j "$job" -t RUNNING -o %N)
  [[ -n $node ]] || { echo "job $job is not running" >&2; return 1; }
  node=$(scontrol show hostnames "$node" | head -1)
  # stdin (an eval's tarball) and stdout (its bundle) pass through this ssh.
  exec ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$node" \
    "AR_JOBID=$job AR_MIN_JOB_MINUTES=${AR_MIN_JOB_MINUTES:-} bash $(printf '%q ' "$D/bin/remote.sh" "$@")"
}

site_check() {  # refuse to start when the job would end mid-run
  local left min=${AR_MIN_JOB_MINUTES:-30} d=0 h=0 m=0
  left=$(squeue -h -j "$AR_JOBID" -o %L)
  # %L is [D-]HH:MM:SS, MM:SS or UNLIMITED.
  [[ $left == UNLIMITED ]] && return 0
  [[ $left == *-* ]] && { d=${left%%-*}; left=${left#*-}; }
  case $(tr -cd : <<<"$left") in
    ::) IFS=: read -r h m _ <<<"$left" ;;
    :) IFS=: read -r m _ <<<"$left" ;;
    *) echo "cannot read the time left in job $AR_JOBID" >&2; return 1 ;;
  esac
  left=$(( (10#$d * 24 + 10#$h) * 60 + 10#$m ))
  (( left >= min )) && return 0
  echo "job $AR_JOBID has $left min left; a run needs $min (AR_MIN_JOB_MINUTES). Start a new job." >&2
  return 1
}

site_env() {
  declare -F module > /dev/null || source /opt/cray/pe/lmod/lmod/init/bash
  # The modules of scripts/hpc_profiles/setonix-gpu.profile, minus its ROCm.
  {
    source /opt/cray/pe/cpe/25.03/restore_lmod_system_defaults.sh
    module load cpe/25.03 pawseyenv/2025.08 PrgEnv-cray craype-x86-trento cray-mpich cce/19.0.0 \
      hdf5/1.14.5-parallel-api-v112
  } > /dev/null 2>&1
  export PYTHONNOUSERSITE=1 TMPDIR=$MYSCRATCH/ar-quokka/tmp XDG_CACHE_HOME=$MYSCRATCH/ar-quokka/cache
  ROCM=$(PYTHONNOUSERSITE=1 "$VENV/bin/rocm-sdk" path --root) || { echo "no ROCm in $VENV (see README.md)" >&2; exit 1; }
  ROCPROF=$ROCM/bin/rocprofv3
  # Inside the job, amd-smi sees only the job's GCD: its clocks, and any
  # other process of yours using it.
  AMDSMI=$ROCM/bin/amd-smi
  # cmake and ninja come from pip too: Quokka needs CMake 3.31+ (policy
  # CMP0179) to keep the HIP device link from seeing libamrex twice.
  export PATH=$VENV/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib:$CRAY_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}
  # Quokka forces AMReX_MPI=ON. These flags let the plain compiler find
  # cray-mpich, as the profile does; the system gcc is too old for C++20.
  local flags="-I$MPICH_DIR/include --gcc-install-dir=/usr/lib64/gcc/x86_64-suse-linux/14"
  export CFLAGS=$flags CXXFLAGS=$flags LDFLAGS="-L$MPICH_DIR/lib -lmpi"
  # One rank never sends device buffers, and the GTL library is built for the
  # system ROCm, not this one.
  export MPICH_GPU_SUPPORT_ENABLED=0
  BUILD_JOBS=$(nproc)
  CONFIGURE_ARGS=(-DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++"
    -DCMAKE_PREFIX_PATH="$ROCM" -DHDF5_ROOT="$PAWSEY_HDF5_HOME")
}
