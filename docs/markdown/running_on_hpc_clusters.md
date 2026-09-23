# Running on HPC clusters

Instructions for running on various HPC clusters are given below.

## Gadi (NCI Australia)

The recommended build procedure on Gadi is:

    source scripts/hpc_profiles/gadi_hopper.profile
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=CUDA -DAMReX_SPACEDIM=3
    cmake --build build -j 8 --target HydroBlast3D

Then a single-node test job can be run with:

    qsub scripts/pbs/gpuhopper-1node.pbs

You can replace `HydroBlast3D` with the name of the problem you want to compile.

### Using VisIt

You can use VisIt in client/server mode with the following server-side [patch for the launcher script](https://gist.github.com/BenWibking/15645ff90819f2808fdb7a04b50b4a1e).

A host file is provided [here](https://gist.github.com/BenWibking/5fa4d6d419dd0adf5da0435e5057b335). You must change the username, project code, and server-side VisIt path.

## Setonix (Pawsey)

The recommended build procedure on Setonix is:

    source scripts/hpc_profiles/setonix-gpu.profile
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DAMReX_GPU_BACKEND=HIP -DAMReX_SPACEDIM=3 -DHDF5_ROOT="$PAWSEY_HDF5_HOME"
    cmake --build build -j 32 --target HydroBlast3D

Then a single-node test job can be run with:

    sbatch scripts/slurm/setonix-1node.submit

You can replace `HydroBlast3D` with the name of the problem you want to compile.

### Workaround for interconnect issues

If interconnect issues are observed, it is recommended to add the line :

    export FI_CXI_RX_MATCH_MODE=software

to your job scripts.

## NASA HEC

### Athena
Once you have logged into the front end, log in to Athena:
    ssh afe0X
-where X=1-6.

After loading the modules Quokka can be build using:
    cmake ../..   -DCMAKE_BUILD_TYPE=Release -G Ninja  -DQUOKKA_PYTHON=OFF 

### Cabeus

Cabeus machines are accessed by:
    ssh cfe01

Quokka can be built on GPUs using:
    cmake ../..   -DCMAKE_BUILD_TYPE=Release   -DAMReX_GPU_BACKEND=CUDA   -DAMReX_SPACEDIM=3   -G Ninja  -DQUOKKA_PYTHON=OFF -DHDF5_ROOT=/nasa/hdf5/1.12.2_nvidia_serial -DAMReX_DIFFERENT_COMPILER=ON -DAMReX_CUDA_ARCH=8.0 -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_CUDA_COMPILER=/nasa/nvidia/hpc_sdk/toss4/Linux_x86_64/25.11/compilers/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=g++ 