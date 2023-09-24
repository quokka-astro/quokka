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
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=58880M      #Indicate the amount of memory per node when asking for share resources
#SBATCH --time=01:00:00

# ---
# Load here the needed modules

# ---
# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   #To define the number of threads, in this case will be 32
export OMP_PLACES=cores     #To bind threads to cores
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.

# ---
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
# Run the desired code:
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS -m block:block:block ./src/PopIII/popiii ../tests/PopIII.in
