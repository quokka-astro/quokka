#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
nsys profile --trace=cuda,nvtx,mpi,ucx --nic-metrics=true ./shock_cloud ShockCloud_256.in
