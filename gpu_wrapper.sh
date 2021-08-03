#!/bin/bash

export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
./build/src/test_radhydro3d_shell tests/radhydro_shell.in amrex.async_out=1
