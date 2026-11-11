#!/usr/bin/env bash
# Regenerates the GOW network (src/networks/GOW) from jaff's GOW++ network and this directory's jaffgen.toml, and
# writes its lookup tables, gow_tables.hdf5. The hand-written src/networks/GOW/tables.hpp and tables.cpp are kept.
#
# The quokka template features this network needs (runtime parameters, column densities, table lookups) are on the
# jaff branch james471/quokka-gow, which is not pushed yet, so a local jaff checkout of that branch is required.
set -euo pipefail

usage() {
	echo "Usage: $0 --jaff-path <path-to-jaff-checkout>" >&2
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

[[ -n "${JAFF_PATH}" ]] || usage
if [[ ! -d "${JAFF_PATH}" ]]; then
	echo "error: --jaff-path '${JAFF_PATH}' does not exist or is not a directory" >&2
	exit 1
fi
JAFF_DIR="$(cd "${JAFF_PATH}" && pwd)"

# Pin sympy version so that jaffgen's output is reproducible
SYMPY_VERSION="1.14.0"

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${CONFIG_DIR}/../GOW"
mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

echo "==> Using jaff checkout at ${JAFF_DIR} ($(git -C "${JAFF_DIR}" rev-parse --short HEAD))"
echo "==> Creating venv at ${TMPDIR}/venv"
python3 -m venv "${TMPDIR}/venv"
echo "==> Installing jaff from ${JAFF_DIR}"
"${TMPDIR}/venv/bin/pip" install --quiet -e "${JAFF_DIR}"
echo "==> Pinning sympy==${SYMPY_VERSION}"
"${TMPDIR}/venv/bin/pip" install --quiet "sympy==${SYMPY_VERSION}"

echo "==> Running jaffgen, output to ${OUTDIR}"
"${TMPDIR}/venv/bin/jaffgen" \
	--config "${CONFIG_DIR}/jaffgen.toml" \
	--network "${JAFF_DIR}/networks/GOW++/GOW++.jet" \
	--template quokka \
	--outdir "${OUTDIR}"

echo "==> Writing the lookup tables"
"${TMPDIR}/venv/bin/python" "${CONFIG_DIR}/make_tables.py" \
	--gow-hdf5 "${JAFF_DIR}/networks/GOW++/GOW++.hdf5" \
	--outdir "${OUTDIR}"

echo "==> Running clang-format on generated files"
clang-format -i "${OUTDIR}"/network.hpp "${OUTDIR}"/network.cpp

echo "==> Done"
