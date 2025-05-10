#!/bin/bash
#SBATCH --job-name=timeout_check
#SBATCH --output=job.log
#SBATCH --error=job.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1

NO_OUTPUT_LIMIT=$((5 * 60)) # kill after 5 minutes of no output
JOB_ID=$SLURM_JOB_ID

monitor_output() {
    local logfile="$1"
    while true; do
        python3 inotify_watcher.py "$logfile" $NO_OUTPUT_LIMIT
        local status=$?
        if [ $status -eq 0 ]; then
            # File modified, continue watching
            continue
        else
            echo "No output written for $NO_OUTPUT_LIMIT seconds. Cancelling job $JOB_ID."
            scancel "$JOB_ID"
            exit 1
        fi
    done
}

monitor_output "$SLURM_JOB_OUTPUT" &

# Your main job commands
for i in {1..10}; do
    echo "Output line $i"
    sleep 30
done

wait
