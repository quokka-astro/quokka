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
    local waited=0

    echo "=========================================="
    echo "Checking for conflicting jobs on the GPU"
    echo "=========================================="

    while true; do
        local gpu_procs=""
        gpu_procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
            | grep -v "^0," || true)

        if [ -z "$gpu_procs" ]; then
            echo "✓ GPU is free, proceeding..."
            echo ""
            return
        fi

        local count
        count=$(echo "$gpu_procs" | grep -c '[0-9]' || true)
        local now
        now=$(date +%Y%m%d\ %H%M%S)
        echo "Waiting: GPU has ${count} active CUDA process(es). Rechecking in ${check_interval}s (${waited}s elapsed) [${now}]..."
        sleep "$check_interval"
        waited=$((waited + check_interval))
    done
}

sif="quokka-linux-amd64-cuda-azp-agent-cached.sif"
if [ ! -f "$sif" ]; then
  singularity pull "$sif" docker://ghcr.io/quokka-astro/quokka-linux-amd64-cuda-azp-agent:development
fi

#cd azp-agent-avatargpu

TARGET=/priv/avatar/cche/azp-agent-in-docker-cuda/azp-agent-avatargpu/regression-tests

log="${TARGET}/reg-logs/crontab-reglog-$(date +%Y%m%d_%H%M%S).log"

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
    >"$log" 2>&1

