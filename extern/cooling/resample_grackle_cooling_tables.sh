#!/usr/bin/env bash
# Regenerate all committed resampled cooling HDF5 tables using the Python script.
# Run from any directory; the script changes into its own directory first.
set -euo pipefail

cd "$(dirname "$0")"

python3 resample_grackle_cooling_tables.py \
    --output "CloudyData_UVB=HM2012_resampled.h5"

python3 resample_grackle_cooling_tables.py \
    --exclude_pe \
    --output "CloudyData_UVB=HM2012_resampled_noPE.h5"

python3 resample_grackle_cooling_tables.py \
    --shield \
    --output "CloudyData_UVB=HM2012_shielded_resampled.h5"

python3 resample_grackle_cooling_tables.py \
    --shield --exclude_pe \
    --output "CloudyData_UVB=HM2012_shielded_resampled_noPE.h5"

echo "All tables regenerated."
