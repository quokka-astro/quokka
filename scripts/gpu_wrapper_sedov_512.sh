#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
#hpcrun -e gpu=nvidia --trace ./build/src/test_hydro3d_blast tests/blast_unigrid_512.in amrex.async_out=1
./build/src/HydroBlast3D/test_hydro3d_blast tests/blast_unigrid_512.in amrex.async_out=1 amrex.use_gpu_aware_mpi=1

