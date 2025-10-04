#!/bin/bash
set -x

cmake --build build --target spherical_collapse
cd tests
mpirun -np 8 ../build/src/problems/SphericalCollapse/spherical_collapse ../inputs/SphericalCollapse.in plotfile_interval=10 max_timesteps=0
cd ..
uv run scripts/python/validate_openpmd_particles.py --openpmd tests/plt00000.bp --csv-root tests
