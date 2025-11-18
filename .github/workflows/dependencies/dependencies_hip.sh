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

# Ref.: https://rocmdocs.amd.com/en/latest/deploy/linux/os-native/install.html
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Detect Ubuntu codename and choose a compatible ROCm suite
# ROCm 7.0 provides packages for Ubuntu 22.04 (jammy). Some runners
# (e.g., ubuntu-latest on 24.04 noble) are not yet supported by AMD's repo.
# In such cases, fall back to jammy which is known to work for ROCm 7.0.
if [ -r /etc/os-release ]; then
  . /etc/os-release
fi

detected_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
rocm_suite="jammy"
if [ "${detected_codename}" = "jammy" ]; then
  rocm_suite="jammy"
else
  # Map all other codenames (e.g., noble, focal, etc.) to jammy for ROCm 7.0
  rocm_suite="jammy"
fi

for ver in 7.0; do
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ver ${rocm_suite} main" \
      | sudo tee --append /etc/apt/sources.list.d/rocm.list
done

echo 'export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin:$PATH' \
  | sudo tee -a /etc/profile.d/rocm.sh
# we should not need to export HIP_PATH=/opt/rocm/hip with those installs

sudo apt-get update

# Ref.: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#installing-development-packages-for-cross-compilation
# meta-package: rocm-dkms
# OpenCL: rocm-opencl
# other: rocm-dev rocm-utils
sudo apt-get install -y --no-install-recommends \
    build-essential \
    libc++-dev      \
    libc++abi-dev   \
    gfortran        \
    libnuma-dev     \
    libopenmpi-dev  \
    openmpi-bin     \
    rocm-dev7.0.0        \
    roctracer-dev7.0.0   \
    rocprofiler-dev7.0.0 \
    rocrand-dev7.0.0     \
    hiprand-dev7.0.0     \
    rocprim-dev7.0.0     \
    rocsparse-dev7.0.0

# activate
#
source /etc/profile.d/rocm.sh
hipcc --version
hipconfig --full
which clang
which clang++
which flang
