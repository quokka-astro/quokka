#!/bin/bash

set -eu

export PATH="/opt/singularity-ce/4.3.0/bin:$PATH"

sif="quokka-linux-amd64-cuda-azp-agent-cached.sif"
if [ ! -f "$sif" ]; then
  singularity pull "$sif" docker://ghcr.io/quokka-astro/quokka-linux-amd64-cuda-azp-agent:development
fi

#cd azp-agent-avatargpu

TARGET=/priv/avatar/cche/azp-agent-in-docker-cuda/azp-agent-avatargpu/regression-tests

log="${TARGET}/reg-logs/crontab-reglog-$(date +%Y%m%d_%H%M%S).log"

singularity exec --nv \
    --bind $TARGET:$TARGET \
    --pwd $TARGET \
    $sif \
    bash quokka2/scripts/bash/run-regression-tests.sh --ini-file ${TARGET}/quokka/regression/quokka-tests.ini \
    --ccache-dir ${TARGET}/ccache --source-dir ${TARGET}/quokka \
    >"$log" 2>&1

