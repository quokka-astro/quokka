#!/bin/bash

set -eu

export PATH="/opt/singularity-ce/4.3.0/bin:$PATH"

#######################################
# Wait for GPU to be free.
# Run on the HOST before launching Singularity so that GPU memory
# allocated by the container runtime itself does not cause false alarms.
#######################################
wait_for_gpu() {
    local check_interval=60
    local max_intervals=240
    local waited=0

    echo "=========================================="
    echo "Checking for conflicting jobs on the GPU"
    echo "=========================================="

    local intervals=0
    while [ "$intervals" -lt "$max_intervals" ]; do
        local gpu_procs=""
        gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
            | grep -v "^0," || true)

        # Calculate total GPU memory usage
        local total_memory=0
        if [ -n "$gpu_procs" ]; then
            total_memory=$(echo "$gpu_procs" | awk -F',' '{sum += $2} END {print int(sum)}')
        fi

        # Consider GPU free if memory usage is under 6000 MB
        if [ "$total_memory" -le 6000 ]; then
            echo "✓ GPU is free (${total_memory} MB used), proceeding..."
            echo ""
            return
        fi

        local count
        count=$(echo "$gpu_procs" | grep -c '[0-9]' || true)
        local now
        now=$(date +%Y%m%d\ %H%M%S)
        echo "Waiting: GPU has ${count} active CUDA process(es) using ${total_memory} MB (threshold: 6000 MB). Rechecking in ${check_interval}s (${waited}s elapsed) [${now}]..."
        sleep "$check_interval"
        waited=$((waited + check_interval))
        intervals=$((intervals + 1))
    done

    echo "Timed out waiting for GPU after ${waited}s. Aborting."
    echo ""
    exit 1
}

sif="quokka-linux-amd64-cuda-azp-agent-cached.sif"
if [ ! -f "$sif" ]; then
  singularity pull "$sif" docker://ghcr.io/quokka-astro/quokka-linux-amd64-cuda-azp-agent:development
fi

#cd azp-agent-avatargpu

TARGET=/priv/avatar/cche/azp-agent-in-docker-cuda/azp-agent-avatargpu/regression-tests

# Check GPU occupancy on the host before launching the container.
# This avoids false alarms from GPU memory allocated by Singularity's --nv init.
wait_for_gpu

# 4 hours timeout
timeout 14400 singularity exec --nv \
    --bind $TARGET:$TARGET \
    --pwd $TARGET \
    $sif \
    bash quokka2/scripts/bash/run-regression-tests.sh --ini-file ${TARGET}/quokka/regression/quokka-tests.ini \
    --ccache-dir ${TARGET}/ccache --source-dir ${TARGET}/quokka \
    --skip-gpu-wait \
    || {
        rc=$?
        if [ $rc -eq 124 ]; then
            echo "ERROR: singularity exec timed out after 4 hours."
        fi
        exit $rc
    }

