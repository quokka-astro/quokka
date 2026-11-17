#!/usr/bin/env bash
#
# Copyright 2020 The AMReX Community
#
# License: BSD-3-Clause-LBNL
# Authors: Axel Huebl

# search recursive inside a folder if a file contains tabs
#
# @result 0 if no files are found, else 1
#

set -eu -o pipefail

# `man apt.conf`:
#   Number of retries to perform. If this is non-zero APT will retry
#   failed files the given number of times.
echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries

# Ref.: https://rocm.docs.amd.com/en/latest/install/rocm.html
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg > /dev/null

# AMD names each ROCm repository after the Ubuntu release, e.g. ubuntu2404 for 24.04.
. /etc/os-release
rocm_distribution="ubuntu${VERSION_ID//./}"
rocm_repository="https://repo.amd.com/rocm/packages-multi-arch/${rocm_distribution}"
if ! curl --fail --silent --head "${rocm_repository}/dists/stable/Release" > /dev/null; then
  echo "error: AMD has not published a ROCm repository for Ubuntu ${VERSION_ID} (${rocm_distribution}); keep this job on the previous Ubuntu release until it does" >&2
  exit 1
fi

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] ${rocm_repository} stable main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list

echo 'export ROCM_PATH=/opt/rocm/core-7.14' | sudo tee /etc/profile.d/rocm.sh
echo "export PATH=\$ROCM_PATH/lib/llvm/bin:\$ROCM_PATH/bin:\$PATH" \
  | sudo tee -a /etc/profile.d/rocm.sh
echo "export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}" \
  | sudo tee -a /etc/profile.d/rocm.sh

sudo apt-get update

sudo apt-get install -y --no-install-recommends \
    build-essential \
    libc++-dev      \
    libc++abi-dev   \
    gfortran        \
    libnuma-dev     \
    libopenmpi-dev  \
    openmpi-bin     \
    amdrocm-core-dev7.14-gfx90a

# activate
#
source /etc/profile.d/rocm.sh
hipcc --version
hipconfig --full
which clang
which clang++
which flang
