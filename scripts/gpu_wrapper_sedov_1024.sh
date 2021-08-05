#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
./build/src/test_hydro3d_blast tests/blast_amr_1024.in amrex.async_out=1
