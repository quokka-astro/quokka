#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
./shock_cloud ShockCloud_256.in amrex.async_out=1 amrex.abort_on_out_of_gpu_memory=1
