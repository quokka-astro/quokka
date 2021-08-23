#!/bin/bash

export CUDA_VISIBLE_DEVICES=$(($OMPI_COMM_WORLD_LOCAL_RANK % 4))
hpcrun -e gpu=nvidia --trace ./build/src/test_hydro3d_blast tests/blast_unigrid_1024.in
#./build/src/test_hydro3d_blast tests/blast_unigrid_1024.in
