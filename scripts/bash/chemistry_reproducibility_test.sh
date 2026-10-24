#!/bin/bash
#
# chemistry_reproducibility_test.sh: check that every CHEMISTRY and PHOTOCHEMISTRY
# test problem reproduces its reference results bit for bit.
#
# Each case runs for STEPS timesteps on one MPI rank (problems that are 64^3 under
# ctest run on a 16^3 grid) and writes a plotfile at step 0
# and at step STEPS. scripts/python/plotfile_hashes.py fingerprints the plotfiles
# (SHA-256 of every file, plus SHA-256, min, max and sum of every variable), and the
# fingerprint, together with the exit status of the run, must match the reference
# in extern/chemistry_reproducibility/ exactly. The exit status is part of the
# fingerprint rather than a pass/fail criterion because many of these problems
# compare against an analytic solution at the end of the run, which is expected
# to fail after only STEPS steps.
#
# The references are only valid for the toolchain they were made with (see
# extern/chemistry_reproducibility/toolchain.txt); a different compiler, libc,
# architecture or set of flags changes the results. --generate makes new ones.
#
# PopIII's initial velocity perturbations are part of the references too
# (PopIII_zdrv.hdf5): src/turbulence/perturbation.py makes them in --generate mode,
# and the check reuses that file. numpy's float64 power differs in the last bit
# between CPUs with and without AVX-512, so the script's output depends on which
# runner it lands on.
#
# Usage:
#   chemistry_reproducibility_test.sh --build-dir DIR [--generate] [options]
#   chemistry_reproducibility_test.sh --list-targets
#
# Options:
#   --build-dir DIR      CMake build directory holding the executables (required)
#   --root DIR           source tree whose inputs/ and tests/ are used
#                        (default: the tree holding this script)
#   --reference-dir DIR  reference store (default: ROOT/extern/chemistry_reproducibility)
#   --work-dir DIR       where runs are written (default: ROOT/tests/chemistry_reproducibility)
#   --cases "A B ..."    run only these cases
#   --generate           write new references instead of checking
#   --list-targets       print the CMake targets the cases need, one per line, and exit

set -euo pipefail

STEPS=10

# The 64^3 problems run on a 16^3 grid: the check is about the code path, not the
# resolution, and this keeps the CPU run short.
SMALL_GRID="amr.n_cell=16 16 16 amr.blocking_factor=8"

# DTypeFrontVC is left out: it never calls evolve() and writes no plotfile after its
# burn, so a fingerprint would only cover its initial conditions.
#
# <case name>:<problem directory>:<executable>:<input file in inputs/>:<extra arguments>
CASES=(
	"OneZonePhotoionization:OneZonePhotoionization:OneZonePhotoionization:OneZonePhotoionization.toml:"
	"OneZonePhotoionizationWithEnergy:OneZonePhotoionization:OneZonePhotoionization:OneZonePhotoionizationWithEnergy.toml:"
	"PrimordialChem:PrimordialChem:PrimordialChem:PrimordialChem.toml:"
	"PopIII:PopIII:PopIII:PopIII.toml:${SMALL_GRID}"
	"DTypeFront1D:DTypeFront1D:DTypeFront1D:DTypeFront1D.toml:"
	"StromgrenSphere:StromgrenSphere:StromgrenSphere:StromgrenSphere.toml:"
	"StromgrenSphereTempIndependentRecombination:StromgrenSphere:StromgrenSphere:StromgrenSphereTempIndependentRecombination.toml:"
	"StromgrenSphereRSLA:StromgrenSphereRSLA:StromgrenSphereRSLA:StromgrenSphereRSLA.toml:"
	"DTypeFront_JAFF:DTypeFront_JAFF:DTypeFront_JAFF:DTypeFront_JAFF.toml:${SMALL_GRID}"
	"DTypeFrontRadPres:DTypeFront_JAFF:DTypeFront_JAFF:DTypeFrontRadPres.toml:${SMALL_GRID}"
	"DTypeFront3D:DTypeFront3D:DTypeFront3D:DTypeFront3D.toml:${SMALL_GRID}"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASHER="${SCRIPT_DIR}/../python/plotfile_hashes.py"

ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR=""
REFERENCE_DIR=""
WORK_DIR=""
SELECTED=""
MODE=check

die() {
	echo "Error: $*" >&2
	exit 2
}

while [ $# -gt 0 ]; do
	case "$1" in
	--build-dir) BUILD_DIR="$2"; shift 2 ;;
	--root) ROOT="$(cd "$2" && pwd)"; shift 2 ;;
	--reference-dir) REFERENCE_DIR="$2"; shift 2 ;;
	--work-dir) WORK_DIR="$2"; shift 2 ;;
	--cases) SELECTED="$2"; shift 2 ;;
	--generate) MODE=generate; shift ;;
	--list-targets)
		for entry in "${CASES[@]}"; do
			IFS=: read -r _ _ exe _ <<<"$entry"
			echo "$exe"
		done | sort -u
		exit 0
		;;
	-h | --help) sed -n '2,/^$/s/^# \{0,1\}//p' "$0"; exit 0 ;;
	*) die "unknown argument: $1" ;;
	esac
done

[ -n "$BUILD_DIR" ] || die "--build-dir is required"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
REFERENCE_DIR="${REFERENCE_DIR:-${ROOT}/extern/chemistry_reproducibility}"
WORK_DIR="${WORK_DIR:-${ROOT}/tests/chemistry_reproducibility}"
mkdir -p "$REFERENCE_DIR" "$WORK_DIR"
REFERENCE_DIR="$(cd "$REFERENCE_DIR" && pwd)"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if [ -n "$SELECTED" ]; then
	for name in $SELECTED; do
		printf '%s\n' "${CASES[@]}" | grep -q "^${name}:" || die "unknown case: $name"
	done
fi

is_selected() {
	[ -z "$SELECTED" ] || [[ " $SELECTED " == *" $1 "* ]]
}

# Everything that changes floating-point results without changing the source.
toolchain_fingerprint() {
	local cache="${BUILD_DIR}/CMakeCache.txt" cxx key
	echo "arch $(uname -m)"
	echo "os $(uname -s)"
	if ldd --version >/dev/null 2>&1; then
		echo "libc $(ldd --version 2>&1 | head -n 1)"
	fi
	cxx="$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$cache")"
	echo "cxx ${cxx}: $("$cxx" --version | head -n 1)"
	for key in CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE AMReX_SPACEDIM AMReX_PRECISION AMReX_MPI AMReX_OMP AMReX_GPU_BACKEND; do
		echo "cmake ${key}=$(sed -n "s/^${key}:[A-Z]*=//p" "$cache")"
	done
	# numpy computes the fingerprint statistics; numpy and h5py write the PopIII initial
	# perturbations in --generate mode.
	echo "python $(python3 -I -c 'import sys, numpy; print(sys.version.split()[0], "numpy", numpy.__version__)')"
	echo "h5py $(python3 -I -c 'import h5py; print(h5py.__version__)' 2>/dev/null || echo none)"
}

# The CPU is not part of the toolchain fingerprint: Quokka's results do not depend on it
# (it is built without -march), but numpy's do, so the log records it for diagnosis.
cpu_description() {
	if [ -r /proc/cpuinfo ]; then
		echo "$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -n 1), avx512f $(grep -q -w avx512f /proc/cpuinfo && echo yes || echo no)"
	else
		sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m
	fi
}

# make_perturbations <extra arguments>
# PopIII reads its initial velocity perturbations from tests/zdrv.hdf5, which ctest
# makes with the ComputePerturbations fixture. These are the same arguments
# (src/problems/PopIII/CMakeLists.txt), with the box size taken from the amr.n_cell
# override if there is one and from the input file otherwise. Only --generate runs
# this; the check copies the file stored with the references.
make_perturbations() {
	local size
	size="$(echo "$1" | sed -n 's/.*amr\.n_cell=\([0-9][0-9]*\).*/\1/p')"
	if [ -z "$size" ]; then
		size="$(grep '^amr\.n_cell' "${ROOT}/inputs/PopIII.toml" | grep -o '[0-9][0-9]*' | head -n 1)"
	fi
	python3 -I "${ROOT}/src/turbulence/perturbation.py" --kmin=2 --kmax=$((size / 2)) --size="$size" \
		--alpha=1.8 --f_solenoidal=0.66667
}

# run_case <case> <problem dir> <executable> <input> <extra arguments> <run dir>
# Runs one case in ROOT/tests (the inputs use paths relative to it, as under ctest)
# and writes <run dir>/fingerprint.txt.
run_case() {
	local name="$1" dir="$2" exe="$3" input="$4" extra="$5" run_dir="$6" status start plt
	local -a extra_args=()
	read -r -a extra_args <<<"$extra"
	rm -rf "$run_dir"
	mkdir -p "$run_dir"
	start="$(date +%s)"
	set +e
	(
		cd "${ROOT}/tests" || exit 1
		if [ "$name" = PopIII ]; then
			rm -f zdrv.hdf5
			if [ "$MODE" = generate ]; then
				make_perturbations "$extra" || exit 1
				cp zdrv.hdf5 "${REFERENCE_DIR}/PopIII_zdrv.hdf5" || exit 1
			else
				cp "${REFERENCE_DIR}/PopIII_zdrv.hdf5" zdrv.hdf5 || exit 1
			fi
		fi
		"${BUILD_DIR}/src/problems/${dir}/${exe}" "../inputs/${input}" ${extra_args[@]+"${extra_args[@]}"} \
			max_timesteps="$STEPS" plotfile_interval="$STEPS" plotfile_prefix="${run_dir}/plt"
	) >"${run_dir}/log.txt" 2>&1
	status=$?
	set -e
	echo "$(($(date +%s) - start))" >"${run_dir}/seconds"

	{
		echo "case ${name}"
		echo "exit_status ${status}"
		if [ "$name" = PopIII ]; then
			echo "input zdrv.hdf5 $(python3 -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${ROOT}/tests/zdrv.hdf5" 2>/dev/null || echo missing)"
		fi
		local found=0
		for plt in "${run_dir}"/plt*; do
			[ -d "$plt" ] || continue
			found=1
			python3 -I "$HASHER" "$plt"
		done
		if [ "$found" = 0 ]; then
			echo "plotfiles none"
		fi
	} >"${run_dir}/fingerprint.txt"
}

declare -a SUMMARY=()
failed=0
echo "cpu: $(cpu_description)"

if [ "$MODE" = generate ]; then
	for entry in "${CASES[@]}"; do
		IFS=: read -r name dir exe input extra <<<"$entry"
		is_selected "$name" || continue
		echo "=== ${name}"
		run_case "$name" "$dir" "$exe" "$input" "$extra" "${WORK_DIR}/${name}"
		cp "${WORK_DIR}/${name}/fingerprint.txt" "${REFERENCE_DIR}/${name}.txt"
		SUMMARY+=("${name} GENERATED exit_status=$(sed -n 's/^exit_status //p' "${WORK_DIR}/${name}/fingerprint.txt") seconds=$(cat "${WORK_DIR}/${name}/seconds")")
	done

	toolchain_fingerprint >"${REFERENCE_DIR}/toolchain.txt"
	{
		echo "source_commit $(git -C "$ROOT" rev-parse HEAD)"
		echo "source_parent $(git -C "$ROOT" rev-parse HEAD^ 2>/dev/null || echo none)"
		echo "source_subject $(git -C "$ROOT" log -1 --format=%s)"
		echo "microphysics $(git -C "$ROOT" rev-parse HEAD:extern/Microphysics)"
		echo "amrex $(git -C "$ROOT" rev-parse HEAD:extern/amrex)"
		echo "generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "runner_image ${ImageOS:-unknown} ${ImageVersion:-unknown}"
		echo "cpu $(cpu_description)"
		if [ -n "${GITHUB_RUN_ID:-}" ]; then
			echo "github_run ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
		fi
		echo "steps ${STEPS}"
		printf '%s\n' "${SUMMARY[@]}"
	} >"${REFERENCE_DIR}/MANIFEST"
else
	if ! toolchain_fingerprint | cmp -s - "${REFERENCE_DIR}/toolchain.txt"; then
		echo "WARNING: this toolchain differs from the one the references were made with, so"
		echo "mismatches below may come from the toolchain rather than from the code:"
		toolchain_fingerprint | diff -u "${REFERENCE_DIR}/toolchain.txt" - || true
	fi
	for entry in "${CASES[@]}"; do
		IFS=: read -r name dir exe input extra <<<"$entry"
		is_selected "$name" || continue
		echo "=== ${name}"
		run_case "$name" "$dir" "$exe" "$input" "$extra" "${WORK_DIR}/${name}"
		if [ ! -f "${REFERENCE_DIR}/${name}.txt" ]; then
			echo "${name}: no reference at ${REFERENCE_DIR}/${name}.txt"
			SUMMARY+=("${name} FAIL (no reference)")
			failed=1
		elif cmp -s "${REFERENCE_DIR}/${name}.txt" "${WORK_DIR}/${name}/fingerprint.txt"; then
			SUMMARY+=("${name} SUCCESS seconds=$(cat "${WORK_DIR}/${name}/seconds")")
		else
			echo "${name}: differs from the reference (- reference, + this run):"
			diff -u "${REFERENCE_DIR}/${name}.txt" "${WORK_DIR}/${name}/fingerprint.txt" | head -n 60 || true
			SUMMARY+=("${name} FAIL")
			failed=1
		fi
	done
fi

echo
echo "=== Summary (${MODE}, ${STEPS} steps)"
printf '%s\n' "${SUMMARY[@]}"
exit "$failed"
