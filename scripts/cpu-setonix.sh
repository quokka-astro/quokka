#!/bin/bash --login

# SLURM directives
#
# Here we specify to SLURM we want an OpenMP job with 32 threads
# a wall-clock time limit of one hour.
#
# Replace [your-project] with the appropriate project name
# following --account (e.g., --account=project123).

#SBATCH --account=pawsey0807
#SBATCH --partition=work
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# ---
# Load here the needed modules

# ---
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
# Run the desired code:
srun ./src/PopIII/popiii ../tests/PopIII.in
