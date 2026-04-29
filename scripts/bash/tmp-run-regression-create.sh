#!/bin/bash

set -e

module load singularity-ce

sif="quokka-linux-amd64-cuda-azp-agent-cached.sif"
if [ ! -f "$sif" ]; then
  singularity pull "$sif" docker://ghcr.io/quokka-astro/quokka-linux-amd64-cuda-azp-agent:development
fi

cd azp-agent-avatargpu

pwd

TARGET=/priv/avatar/cche/azp-agent-in-docker-cuda/azp-agent-avatargpu
#REGTEST=regression-tests/quokka/extern/regression_testing/regtest.py
REGTEST=regression-tests/regression_testing/regtest.py

# Default title to updateYYYYMMDD
TITLE="update$(date +%Y%m%d)"
JOBS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --title)
            TITLE="$2"
            shift 2
            ;;
        *)
            JOBS+=("$1")
            shift
            ;;
    esac
done

# Check if we have jobs
if [ ${#JOBS[@]} -eq 0 ]; then
    echo "Usage: $0 [--title TITLE] <job1> [job2 ...]"
    exit 1
fi

# Determine which flag to use based on number of job arguments
if [ ${#JOBS[@]} -eq 1 ]; then
    singularity exec --nv \
        --bind $TARGET:$TARGET \
        --pwd $TARGET \
        ../$sif python3 $REGTEST \
        --make_benchmarks "$TITLE" \
        --single_test "${JOBS[0]}" \
        --source_branch "chong/reg-test-tstop" \
        regression-tests/quokka/regression/quokka-tests.ini
else
    # Join jobs into a single space-separated string
    JOBS_STR="${JOBS[*]}"
    singularity exec --nv \
        --bind $TARGET:$TARGET \
        --pwd $TARGET \
        ../$sif python3 $REGTEST \
        --make_benchmarks "$TITLE" \
        --tests "$JOBS_STR" \
        --source_branch "chong/reg-test-tstop" \
        regression-tests/quokka/regression/quokka-tests.ini
fi

#singtest="RandomBlastAMR-GPU"
# cmd='--make_benchmarks "$singtest" --single_test "$singtest" regression-tests/quokka-generate-benchmarks/quokka-tests.ini --source_pr 1547'
# --single_test "$singtest" \

# ../$sif python3 $REGTEST --keep_build --reuse_build --build_name "build" --make_benchmarks "all" \

