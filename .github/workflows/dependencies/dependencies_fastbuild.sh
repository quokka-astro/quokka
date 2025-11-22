#!/usr/bin/env bash
#
# Install FASTBuild.
#  - Linux: download the upstream binary release.
#  - macOS: rely on Homebrew.

set -euo pipefail

if [[ $# -eq 1 ]]; then
  VERSION=$1
else
  VERSION=1.15
fi

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

if command -v fbuild >/dev/null 2>&1; then
  echo "fastbuild is already installed"
  exit 0
fi

case "${OS_NAME}" in
  Linux)
    # Install unzip if not present
    sudo apt-get update && sudo apt-get install -y unzip wget

    # FASTBuild only provides x64 binaries for Linux
    if [[ "${ARCH_NAME}" != "x86_64" ]]; then
        echo "Error: FASTBuild only provides pre-compiled binaries for Linux x86_64. You are on ${ARCH_NAME}."
        exit 1
    fi

    URL="https://www.fastbuild.org/downloads/v${VERSION}/FASTBuild-Linux-x64-v${VERSION}.zip"
    echo "Downloading FASTBuild from ${URL}..."
    wget -q "${URL}" -O fastbuild.zip
    unzip -q fastbuild.zip
    
    # Find the binary
    FOUND=$(find . -name fbuild -type f | head -n 1)
    if [[ -n "$FOUND" ]]; then
        sudo cp "$FOUND" /usr/local/bin/fbuild
        sudo chmod +x /usr/local/bin/fbuild
        echo "FASTBuild installed to /usr/local/bin/fbuild"
    else
        echo "Error: Could not find fbuild binary in zip"
        exit 1
    fi
    
    rm -rf fastbuild.zip "FASTBuild-Linux-x64-v${VERSION}"
    ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      brew install fastbuild
    else
      echo "error: Homebrew is required to install fastbuild on macOS" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unsupported platform ${OS_NAME} for fastbuild installation" >&2
    exit 1
    ;;
esac
