#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
EXE="${EXE:-}"
FCOMPARE_DIR="${ROOT_DIR}/extern/amrex/Tools/Plotfile"
INPUT_FILE="${ROOT_DIR}/inputs/SN_feedback_only.in"
PARTICLE_FILE="${ROOT_DIR}/inputs/SN.txt"
PLOTBOOST_EXE="${PLOTBOOST_EXE:-}"

run_sn_case() {
  local run_dir="$1"
  shift
  local boost_args=("$@")

  mkdir -p "${run_dir}"
  echo "Running: ${EXE} ${INPUT_FILE} plotfile_prefix=${run_dir}/plt max_timesteps=1 ${boost_args[*]}"
  set +e
  "${EXE}" "${INPUT_FILE}" \
    problem.SN_particles_file="${PARTICLE_FILE}" \
    plotfile_interval=1 \
    plotfile_prefix="${run_dir}/plt" \
    max_timesteps=1 \
    "${boost_args[@]}"
  local exit_code=$?
  set -e
  if [ ${exit_code} -ne 0 ]; then
    echo "Warning: SN run exited with code ${exit_code}"
  fi
}

find_latest_plotfile() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type d -name "plt*" | sort | tail -1
}

if [ -z "${EXE}" ]; then
  EXE="${BUILD_DIR}/src/problems/SN/SN"
fi

if [ ! -x "${EXE}" ]; then
  echo "SN executable not found at ${EXE}. Building SN target..."
  if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "Build directory not configured. Running cmake configure..."
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3
  fi
  cmake --build "${BUILD_DIR}" --target SN
fi

if [ ! -x "${EXE}" ]; then
  EXE_FOUND=$(find "${BUILD_DIR}" -type f -perm -111 -name "SN" 2>/dev/null | head -1 || true)
  if [ -n "${EXE_FOUND}" ]; then
    EXE="${EXE_FOUND}"
  else
    echo "Error: SN executable still not found at ${EXE} after build."
    exit 1
  fi
fi

FCOMPARE=$(find "${FCOMPARE_DIR}" -name "fcompare.*.ex" 2>/dev/null | head -1 || true)
if [ -z "${FCOMPARE}" ] || [ ! -f "${FCOMPARE}" ]; then
  echo "fcompare tool not found, building it..."
  (cd "${FCOMPARE_DIR}" && make -j"$(nproc)" programs=fcompare)
  FCOMPARE=$(find "${FCOMPARE_DIR}" -name "fcompare.*.ex" 2>/dev/null | head -1 || true)
fi

if [ -z "${FCOMPARE}" ] || [ ! -f "${FCOMPARE}" ]; then
  echo "Error: fcompare binary not found after build."
  exit 1
fi

if [ -z "${PLOTBOOST_EXE}" ]; then
  PLOTBOOST_EXE="${BUILD_DIR}/plotfile_boost"
fi

if [ ! -x "${PLOTBOOST_EXE}" ]; then
  echo "plotfile_boost executable not found at ${PLOTBOOST_EXE}. Building plotfile_boost target..."
  if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "Build directory not configured. Running cmake configure..."
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DAMReX_SPACEDIM=3
  fi
  cmake --build "${BUILD_DIR}" --target plotfile_boost
fi

if [ ! -x "${PLOTBOOST_EXE}" ]; then
  PLOTBOOST_FOUND=$(find "${BUILD_DIR}" -type f -perm -111 -name "plotfile_boost" 2>/dev/null | head -1 || true)
  if [ -n "${PLOTBOOST_FOUND}" ]; then
    PLOTBOOST_EXE="${PLOTBOOST_FOUND}"
  else
    echo "Error: plotfile_boost executable not found after build."
    exit 1
  fi
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

RUN_NO_BOOST="${TMP_DIR}/run_no_boost"
RUN_BOOST="${TMP_DIR}/run_boost"

echo "Running SN feedback-only test without boost..."
run_sn_case "${RUN_NO_BOOST}"

echo "Running SN feedback-only test with boost..."
run_sn_case "${RUN_BOOST}" "problem.boost_velocity=1.0e8 0.0 0.0"

PLOT_NO_BOOST="$(find_latest_plotfile "${RUN_NO_BOOST}")"
PLOT_BOOST="$(find_latest_plotfile "${RUN_BOOST}")"

if [ -z "${PLOT_NO_BOOST}" ] || [ -z "${PLOT_BOOST}" ]; then
  echo "Error: plotfiles not found."
  exit 1
fi

PLOT_BOOST_DEBOOST="${RUN_BOOST}/plt_deboosted"
rm -rf "${PLOT_BOOST_DEBOOST}"

echo "De-boosting boosted plotfile..."
echo "Running: ${PLOTBOOST_EXE} ${PLOT_BOOST} ${PLOT_BOOST_DEBOOST} -1.0e7 0.0 0.0"
"${PLOTBOOST_EXE}" "${PLOT_BOOST}" "${PLOT_BOOST_DEBOOST}" -1.0e7 0.0 0.0

echo "Comparing plotfiles:"
echo "  no boost: ${PLOT_NO_BOOST}"
echo "  deboost:  ${PLOT_BOOST_DEBOOST}"

echo "Running fcompare:"
echo "${FCOMPARE} --abs_tol 0.0 --rel_tol 0.0 ${PLOT_NO_BOOST} ${PLOT_BOOST_DEBOOST}"
FCOMPARE_OUTPUT="$(mktemp)"
"${FCOMPARE}" --abs_tol 0.0 --rel_tol 0.0 "${PLOT_NO_BOOST}" "${PLOT_BOOST_DEBOOST}" 2>&1 | tee "${FCOMPARE_OUTPUT}"
FCOMPARE_EXIT_CODE=${PIPESTATUS[0]}
echo "fcompare exit code: ${FCOMPARE_EXIT_CODE}"
if [ -s "${FCOMPARE_OUTPUT}" ]; then
  echo "fcompare output:"
  cat "${FCOMPARE_OUTPUT}"
else
  echo "fcompare produced no output."
fi
