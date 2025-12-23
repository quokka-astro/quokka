#!/bin/bash

#hpcrun -e gpu=nvidia --trace ./build/src/test_hydro3d_blast tests/blast_unigrid_1024.in
./build/src/HydroBlast3D/test_hydro3d_blast tests/blast_unigrid_1024.in
