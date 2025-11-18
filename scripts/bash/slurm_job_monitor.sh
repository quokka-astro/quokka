#!/bin/bash
#SBATCH --job-name=job_with_monitor
#SBATCH --output=job_output.log
#SBATCH --error=job_output.log
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

LOGFILE="job_output.log"
CHECK_INTERVAL=60     # check every CHECK_INTERVAL seconds
TIMEOUT=300           # timeout without output after TIMEOUT seconds

monitor_log() {
    echo "Monitoring $LOGFILE for hangs..."
    # Get initial line count
    last_lines=$(wc -l < "$LOGFILE" 2>/dev/null || echo 0)
    last_change_time=$(date +%s)

    while true; do
        sleep "$CHECK_INTERVAL"
        current_lines=$(wc -l < "$LOGFILE" 2>/dev/null || echo 0)
        current_time=$(date +%s)

        if (( current_lines > last_lines )); then
            # New output detected
            last_lines=$current_lines
            last_change_time=$current_time
        else
            # No new output detected
            elapsed=$(( current_time - last_change_time ))
            if (( elapsed >= TIMEOUT )); then
                echo "No output in last $TIMEOUT seconds. Canceling job $SLURM_JOB_ID."
                scancel "$SLURM_JOB_ID"
                exit 0
            fi
        fi
    done
}

# Start monitor in background
monitor_log &

# Your main job commands below
for i in {1..5}; do
    echo "Output line $i at $(date)"
    sleep 40
done

