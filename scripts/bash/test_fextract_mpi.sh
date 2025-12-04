#!/bin/bash
# ABOUTME: Run the src/util fextract helper via the dedicated driver with 1 and multiple MPI ranks and compare outputs

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 -p <plotfile> [-v "<var1 var2>"] [-d <dir>] [-c <coord>] [-n <ranks>] [-m <mpi_launcher>] [-a "<mpi_args>"] [-b <build_dir>] [-k]

Options:
  -p <plotfile>        Path to the plotfile directory to slice (required)
  -v "<vars>"          Space-separated variable names to extract (defaults to all)
  -d <dir>             Slice direction (0, 1, or 2). Default: 0
  -c <coord>           Coordinate value along the slice direction (optional; defaults to domain center)
  -n <ranks>           MPI ranks for the parallel run. Must be >= 2. Default: 2
  -m <mpi_launcher>    MPI launcher command (mpirun, mpiexec, srun, etc.). Default: mpirun
  -a "<mpi_args>"      Extra arguments passed to the MPI launcher (default: --oversubscribe for mpirun/mpiexec)
  -b <build_dir>       CMake build directory containing fextract_util_driver. Default: build
  -k                   Keep temporary files (default cleans up)
  -h                   Show this help message

The script builds the fextract_util_driver target, runs it once with a single rank
and once with the requested parallel ranks, then diffs the outputs.
EOF
    return 1
}

plotfile=""
vars=""
direction=0
coord=""
mpi_ranks=2
mpi_launcher="mpirun"
mpi_args=""
build_dir="build"
keep_tmp=false

while getopts ":p:v:d:c:n:m:a:b:kh" opt; do
    case "${opt}" in
        p) plotfile="${OPTARG}" ;;
        v) vars="${OPTARG}" ;;
        d) direction="${OPTARG}" ;;
        c) coord="${OPTARG}" ;;
        n) mpi_ranks="${OPTARG}" ;;
        m) mpi_launcher="${OPTARG}" ;;
        a) mpi_args="${OPTARG}" ;;
        b) build_dir="${OPTARG}" ;;
        k) keep_tmp=true ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

if [[ -z "${plotfile}" ]]; then
    echo "Error: plotfile is required." >&2
    usage
    exit 1
fi

if [[ ! -d "${plotfile}" ]]; then
    echo "Error: plotfile path '${plotfile}' is not a directory." >&2
    exit 1
fi

if ! [[ "${direction}" =~ ^[0-2]$ ]]; then
    echo "Error: direction must be 0, 1, or 2." >&2
    exit 1
fi

if ! [[ "${mpi_ranks}" =~ ^[0-9]+$ ]] || [[ "${mpi_ranks}" -lt 2 ]]; then
    echo "Error: ranks (-n) must be an integer >= 2." >&2
    exit 1
fi

if ! command -v "${mpi_launcher}" >/dev/null 2>&1; then
    echo "Error: MPI launcher '${mpi_launcher}' not found in PATH." >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
plotfile_abs="${plotfile}"
if [[ "${plotfile_abs}" != /* ]]; then
    plotfile_abs="$(pwd)/${plotfile_abs}"
fi
if [[ "${build_dir}" != /* ]]; then
    build_dir="${repo_root}/${build_dir}"
fi
if [[ -z "${mpi_args}" ]] && { [[ "${mpi_launcher}" == "mpirun" ]] || [[ "${mpi_launcher}" == "mpiexec" ]]; }; then
    mpi_args="--host localhost --bind-to none --map-by slot --oversubscribe"
fi

build_path="${build_dir}/src/fextract_util_driver"

echo "Building fextract_util_driver in ${build_dir}..."
if ! cmake --build "${build_dir}" --target fextract_util_driver >/dev/null; then
    echo "Build failed. Re-run with VERBOSE=1 or without redirect to inspect errors." >&2
    exit 1
fi
echo "Build complete."

if [[ ! -x "${build_path}" ]]; then
    echo "Error: built driver not found at ${build_path}" >&2
    exit 1
fi

tmp_root="${TMPDIR:-/tmp}"
tmp_dir="$(mktemp -d "${tmp_root%/}/fextract_mpi_test.XXXXXX")"
cleanup() {
    if [[ "${keep_tmp}" == false && -d "${tmp_dir}" ]]; then
        rm -rf "${tmp_dir}"
    else
        echo "Temporary files kept at ${tmp_dir}"
    fi
    return 0
}
trap cleanup EXIT

single_out="${tmp_dir}/fextract_single.slice"
multi_out="${tmp_dir}/fextract_multi.slice"

echo "Using driver at ${build_path}"
echo "Running single-rank fextract..."
"${mpi_launcher}" ${mpi_args} -np 1 "${build_path}" plotfile="${plotfile_abs}" outfile="${single_out}" dir="${direction}" ${coord:+coord="${coord}"} ${vars:+vars="${vars}"}

echo "Running ${mpi_ranks}-rank fextract..."
"${mpi_launcher}" ${mpi_args} -np "${mpi_ranks}" "${build_path}" plotfile="${plotfile_abs}" outfile="${multi_out}" dir="${direction}" ${coord:+coord="${coord}"} ${vars:+vars="${vars}"}

echo "Comparing outputs..."
if diff -u "${single_out}" "${multi_out}"; then
    echo "Success: fextract output matches between 1 and ${mpi_ranks} ranks."
else
    echo "Failure: fextract outputs differ." >&2
    echo "Single-rank slice: ${single_out}" >&2
    echo "Multi-rank slice:  ${multi_out}" >&2
    exit 2
fi
echo "Done. Temp files at ${tmp_dir}"
