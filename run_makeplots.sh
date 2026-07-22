#!/bin/bash
#SBATCH -A pawsey0807-gpu
#SBATCH -J makeplots
#SBATCH -o makeplots_%j.out
#SBATCH -t 02:00:00
#SBATCH -p gpu-highmem
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gpus-per-node=1

# Redirect all cache-writing activities to your scratch space
export XDG_CACHE_HOME=/scratch/pawsey0807/ecolekodikara/.cache
export MPLCONFIGDIR=/scratch/pawsey0807/ecolekodikara/.config/matplotlib

# Create these directories if they don't exist
mkdir -p $XDG_CACHE_HOME
mkdir -p $MPLCONFIGDIR

# ── Environment ───────────────────────────────────────────────────────────────
source /scratch/pawsey0807/ecolekodikara/mhddisk/bin/activate

# ── Run ──────────────────────────────────────────────────────────────────────
# The highmem partition will provide the memory overhead you need.
srun -n 16 python makeplots_fast.py

echo "Finished: $(date)"