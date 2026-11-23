#!/usr/bin/env bash
set -euo pipefail

CUDA_VERSION="${1:?usage: dependencies_cuda.sh <cuda-version>, e.g. 13.0}"

cuda_version_dashed="${CUDA_VERSION//./-}"
## NVIDIA names each CUDA repository after the Ubuntu release, e.g. ubuntu2404 for 24.04.
# shellcheck source=/dev/null
. /etc/os-release
cuda_distribution="ubuntu${VERSION_ID//./}"
cuda_repository="https://developer.download.nvidia.com/compute/cuda/repos/${cuda_distribution}/x86_64"
if ! curl --fail --silent --head "${cuda_repository}/Release" > /dev/null; then
    echo "error: NVIDIA has not published a CUDA repository for Ubuntu ${VERSION_ID} (${cuda_distribution}); keep this job on the previous Ubuntu release until it does"
    exit 1
fi

echo 'Acquire::Retries "3";' | sudo tee /etc/apt/apt.conf.d/80-retries
cuda_keyring="${RUNNER_TEMP:-/tmp}/cuda-keyring.deb"
curl --fail --silent --show-error --location --output "${cuda_keyring}" "${cuda_repository}/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i "${cuda_keyring}"
sudo apt-get update

if ! apt-cache show "cuda-compiler-${cuda_version_dashed}" > /dev/null 2>&1; then
    available_versions="$(apt-cache pkgnames cuda-compiler- | sed 's/^cuda-compiler-//; s/-/./' | sort -uV | tr '\n' ' ')"
    echo "error: NVIDIA has no CUDA ${CUDA_VERSION} for Ubuntu ${VERSION_ID}; available versions: ${available_versions}"
    exit 1
fi

sudo apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    g++ \
    gfortran \
    gnupg \
    libopenmpi-dev \
    openmpi-bin \
    pkg-config \
    "cuda-command-line-tools-${cuda_version_dashed}" \
    "cuda-compiler-${cuda_version_dashed}" \
    "cuda-cupti-dev-${cuda_version_dashed}" \
    "cuda-minimal-build-${cuda_version_dashed}" \
    "cuda-nvml-dev-${cuda_version_dashed}" \
    "cuda-nvtx-${cuda_version_dashed}" \
    "libcufft-dev-${cuda_version_dashed}" \
    "libcurand-dev-${cuda_version_dashed}" \
    "libcusparse-dev-${cuda_version_dashed}" \
    "libnvjitlink-dev-${cuda_version_dashed}"

sudo ln -s "cuda-${CUDA_VERSION}" /usr/local/cuda
