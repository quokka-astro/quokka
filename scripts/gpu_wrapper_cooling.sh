#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
../build/src/Cooling/test_cooling Cooling.in amrex.async_out=1 amrex.abort_on_out_of_gpu_memory=1


