#!/usr/bin/env bash
#
# Install CMake.
#  - Linux: download the upstream binary release.
#  - macOS: rely on Homebrew.

set -euo pipefail

if [[ $# -eq 1 ]]; then
  VERSION=$1
else
  VERSION=4.2.0
fi

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

case "${OS_NAME}" in
  Linux)
    # Install wget if not present
    sudo apt-get update && sudo apt-get install -y wget

    if [[ "${ARCH_NAME}" == "x86_64" ]]; then
        FILENAME="cmake-${VERSION}-linux-x86_64.sh"
    elif [[ "${ARCH_NAME}" == "aarch64" ]]; then
        FILENAME="cmake-${VERSION}-linux-aarch64.sh"
    else
        echo "Error: Unsupported architecture ${ARCH_NAME} for CMake installation script."
        exit 1
    fi

    URL="https://github.com/Kitware/CMake/releases/download/v${VERSION}/${FILENAME}"
    echo "Downloading CMake ${VERSION} from ${URL}..."
    wget -q "${URL}" -O "${FILENAME}"
    sudo sh "${FILENAME}" --prefix=/usr/local --skip-license
    rm "${FILENAME}"
    ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      brew install cmake || brew upgrade cmake
    else
      echo "error: Homebrew is required to install cmake on macOS" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unsupported platform ${OS_NAME} for cmake installation" >&2
    exit 1
    ;;
esac
