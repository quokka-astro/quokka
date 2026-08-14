#!/usr/bin/env bash
# Regenerates the photoionization_H_caseB network from this directory's
# jaffgen.toml/dtype.jet. By default, clones jaff at the pinned commit into a
# temp dir and runs jaffgen there. Pass --jaff-path to use an existing local
# jaff checkout instead, bypassing the clone.
set -euo pipefail

usage() {
	echo "Usage: $0 [--jaff-path <path-to-jaff-checkout>]" >&2
	exit 1
}

JAFF_PATH=""
while [[ $# -gt 0 ]]; do
	case "$1" in
	--jaff-path)
		[[ $# -ge 2 ]] || usage
		JAFF_PATH="$2"
		shift 2
		;;
	*)
		usage
		;;
	esac
done

JAFF_REPO="https://github.com/jaff-chemistry/jaff.git"
JAFF_COMMIT="116ba9e"

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$(cd "${CONFIG_DIR}/../photoionization_H_caseB" && pwd)"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

if [[ -n "${JAFF_PATH}" ]]; then
	if [[ ! -d "${JAFF_PATH}" ]]; then
		echo "error: --jaff-path '${JAFF_PATH}' does not exist or is not a directory" >&2
		exit 1
	fi
	JAFF_DIR="$(cd "${JAFF_PATH}" && pwd)"
	echo "==> Using existing jaff checkout at ${JAFF_DIR}"
else
	echo "==> Cloning ${JAFF_REPO} into ${TMPDIR}/jaff"
	git clone "${JAFF_REPO}" "${TMPDIR}/jaff"
	echo "==> Checking out commit ${JAFF_COMMIT}"
	if ! git -C "${TMPDIR}/jaff" checkout "${JAFF_COMMIT}"; then
		echo "error: commit '${JAFF_COMMIT}' not found in ${JAFF_REPO}" >&2
		exit 1
	fi
	JAFF_DIR="${TMPDIR}/jaff"
fi

echo "==> Creating venv at ${TMPDIR}/venv"
python3 -m venv "${TMPDIR}/venv"
echo "==> Installing jaff from ${JAFF_DIR}"
"${TMPDIR}/venv/bin/pip" install --quiet -e "${JAFF_DIR}"

echo "==> Running jaffgen, output to ${OUTDIR}"
"${TMPDIR}/venv/bin/jaffgen" \
	--config "${CONFIG_DIR}/jaffgen.toml" \
	--network "${CONFIG_DIR}/dtype.jet" \
	--template microphysics \
	--lang cxx \
	--outdir "${OUTDIR}"

echo "==> Running clang-format on generated files"
clang-format -i "${OUTDIR}"/*.H "${OUTDIR}"/*.cpp

echo "==> Done"
