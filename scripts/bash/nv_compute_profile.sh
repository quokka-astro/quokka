#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
ncu \
    --set full \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*computePhotoChemistry.*' \
    --launch-count 3 \
    --force-overwrite \
    -o ncu_dtypefront_photochemistry \
    ./build/3d-cuda/src/problems/DTypeFront/DTypeFront \
    ./inputs/DTypeFront.toml amrex.the_arena_init_size=0
  
