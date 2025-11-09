#!/usr/bin/env bash
#
# Install ccache following the pattern used by AMReX CI.
#  - Linux x86_64: download the upstream binary release.
#  - Other Linux architectures: fall back to the system package manager.
#  - macOS: rely on Homebrew.

set -euo pipefail

if [[ $# -eq 1 ]]; then
  CVER=$1
else
  CVER=4.8
fi

OS_NAME=$(uname -s)
ARCH_NAME=$(uname -m)

if command -v ccache >/dev/null 2>&1; then
  exit 0
fi

case "${OS_NAME}" in
  Linux)
    if [[ "${ARCH_NAME}" == "x86_64" ]]; then
      wget -q https://github.com/ccache/ccache/releases/download/v${CVER}/ccache-${CVER}-linux-x86_64.tar.xz
      tar xJf ccache-${CVER}-linux-x86_64.tar.xz
      sudo cp -f ccache-${CVER}-linux-x86_64/ccache /usr/local/bin/
      rm -rf ccache-${CVER}-linux-x86_64 ccache-${CVER}-linux-x86_64.tar.xz
    else
      sudo apt-get update
      sudo apt-get install -y ccache
    fi
    ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      brew install ccache
    else
      echo "error: Homebrew is required to install ccache on macOS" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unsupported platform ${OS_NAME} for ccache installation" >&2
    exit 1
    ;;
esac
