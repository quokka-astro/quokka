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

# Detect the Ubuntu release and choose the matching ROCm 7.14 repository.
if [ -r /etc/os-release ]; then
  . /etc/os-release
fi

detected_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
case "${detected_codename}" in
  jammy)
    rocm_distribution="ubuntu2204"
    ;;
  noble)
    rocm_distribution="ubuntu2404"
    ;;
  *)
    echo "Unsupported Ubuntu codename for ROCm 7.14: ${detected_codename}" >&2
    exit 1
    ;;
esac

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/${rocm_distribution} stable main" \
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
