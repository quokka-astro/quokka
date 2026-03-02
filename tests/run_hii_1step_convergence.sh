#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${ROOT_DIR}/build/src/problems/HIIRegion/HIIRegion"
BASE_IN="${ROOT_DIR}/inputs/HIIRegion.in"
PLOT_HELPER="${ROOT_DIR}/scripts/python/plot_hii_fld_6panel.py"

# Defaults (override via env vars or CLI flags below)
LOG_EVERY="${LOG_EVERY:-1}"
RESIDUAL_TOL="${RESIDUAL_TOL:-1e-3}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-1}"
PLOTFILE_INTERVAL="${PLOTFILE_INTERVAL:-1}"
OUTPUT_PLOT="${OUTPUT_PLOT:-${ROOT_DIR}/tests/hii_fld_6panel_16_32_64_1step.png}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs HIIRegion for N=16,32,64 with one hydro timestep by default, then makes a combined plot.

Options:
  --log-every N         Set particles.stromgren_log_every (default: ${LOG_EVERY})
  --residual-tol X      Set particles.stromgren_residual_tol (default: ${RESIDUAL_TOL})
  --max-timesteps N     Set max_timesteps (default: ${MAX_TIMESTEPS})
  --plotfile-interval N Set plotfile_interval (default: ${PLOTFILE_INTERVAL})
  --output PATH         Output PNG path (default: ${OUTPUT_PLOT})
  -h, --help            Show this help text
EOF
}

sed_in_place() {
  local expr="$1"
  local file="$2"
  local tmp

  tmp="$(mktemp "${file}.tmp.XXXXXX")"
  sed "${expr}" "${file}" > "${tmp}"
  mv "${tmp}" "${file}"
}

extract_stromgren_iterations() {
  local log_file="$1"

  awk '
    match($0, /\[iter[[:space:]]*[0-9]+\]/) {
      token = substr($0, RSTART, RLENGTH)
      gsub(/[^0-9]/, "", token)
      iter = token + 0
      if (iter > max_iter) {
        max_iter = iter
      }
      found = 1
    }
    END {
      if (found) {
        print max_iter
      } else {
        print "N/A"
      }
    }
  ' "${log_file}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-every)
      LOG_EVERY="$2"
      shift 2
      ;;
    --residual-tol)
      RESIDUAL_TOL="$2"
      shift 2
      ;;
    --max-timesteps)
      MAX_TIMESTEPS="$2"
      shift 2
      ;;
    --plotfile-interval)
      PLOTFILE_INTERVAL="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PLOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${EXE}" ]]; then
  echo "Error: executable not found: ${EXE}" >&2
  echo "Build it first: cmake --build build --target HIIRegion" >&2
  exit 1
fi

if [[ ! -f "${BASE_IN}" ]]; then
  echo "Error: input file not found: ${BASE_IN}" >&2
  exit 1
fi

if [[ ! -f "${PLOT_HELPER}" ]]; then
  echo "Error: plot helper not found: ${PLOT_HELPER}" >&2
  exit 1
fi

COOLING_FILE="${ROOT_DIR}/extern/cooling/CloudyData_UVB=HM2012_resampled.h5"
if [[ ! -f "${COOLING_FILE}" ]]; then
  echo "Error: cooling table not found: ${COOLING_FILE}" >&2
  exit 1
fi

QH0_FILE="${ROOT_DIR}/extern/stellar_tables/QH0_mist_9to120_quokka_best.h5"
if [[ ! -f "${QH0_FILE}" ]]; then
  echo "Error: QH0 table not found: ${QH0_FILE}" >&2
  exit 1
fi

STARS_FILE="${ROOT_DIR}/inputs/hii_stars.txt"
if [[ ! -f "${STARS_FILE}" ]]; then
  echo "Error: stars file not found: ${STARS_FILE}" >&2
  exit 1
fi

declare -a PLOTFILES=()
declare -a RUN_FAILED_CASES=()
declare -a MISSING_PLOT_CASES=()
for N in 16 32 64; do
  RUN_DIR="${ROOT_DIR}/tests/hii_runs/${N}"
  mkdir -p "${RUN_DIR}"
  cp "${BASE_IN}" "${RUN_DIR}/HIIRegion.in"
  cp "${STARS_FILE}" "${RUN_DIR}/hii_stars.txt"
  rm -rf "${RUN_DIR}"/plt*

  sed_in_place "s|^amr.n_cell = .*|amr.n_cell = ${N} ${N} ${N}|" "${RUN_DIR}/HIIRegion.in"
  sed_in_place "s|^amr.max_grid_size = .*|amr.max_grid_size = ${N}|" "${RUN_DIR}/HIIRegion.in"
  sed_in_place "s|^plotfile_interval = .*|plotfile_interval = ${PLOTFILE_INTERVAL}|" "${RUN_DIR}/HIIRegion.in"
  sed_in_place "s|^cooling.hdf5_data_file = .*|cooling.hdf5_data_file = \"${COOLING_FILE}\"|" "${RUN_DIR}/HIIRegion.in"
  sed_in_place "s|^particles.stromgren_qh0_table_hdf5_file = .*|particles.stromgren_qh0_table_hdf5_file = \"${QH0_FILE}\"|" "${RUN_DIR}/HIIRegion.in"

  echo "Running HIIRegion ${N}^3 ..."
  RUN_LOG="${RUN_DIR}/run_1step.log"
  if (
    cd "${RUN_DIR}"
    "${EXE}" HIIRegion.in \
      suppress_output=1 \
      max_timesteps="${MAX_TIMESTEPS}" \
      plotfile_interval="${PLOTFILE_INTERVAL}" \
      particles.stromgren_log_every="${LOG_EVERY}" \
      particles.stromgren_residual_tol="${RESIDUAL_TOL}" \
      > "${RUN_LOG}" 2>&1
  ); then
    :
  else
    RUN_FAILED_CASES+=("${N}")
    echo "Warning: HIIRegion ${N}^3 failed. See ${RUN_LOG}" >&2
  fi
  if [[ -f "${RUN_LOG}" ]]; then
    stromgren_iters="$(extract_stromgren_iterations "${RUN_LOG}")"
    echo "Stromgren iterations (${N}^3): ${stromgren_iters}"
  else
    echo "Stromgren iterations (${N}^3): N/A (missing log)" >&2
  fi

  latest_plotfile="$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'plt*' | sort | tail -n 1)"
  if [[ -n "${latest_plotfile}" ]]; then
    PLOTFILES+=("${latest_plotfile}")
  else
    MISSING_PLOT_CASES+=("${N}")
    echo "Warning: missing plotfile for ${N}^3 in ${RUN_DIR}." >&2
  fi
done

if [[ ${#RUN_FAILED_CASES[@]} -gt 0 ]]; then
  echo "One or more runs returned non-zero: ${RUN_FAILED_CASES[*]}" >&2
  exit 1
fi

if [[ ${#MISSING_PLOT_CASES[@]} -gt 0 || ${#PLOTFILES[@]} -ne 3 ]]; then
  echo "Skipping combined plot because not all three plotfiles were generated." >&2
  exit 1
fi

echo "Generating combined plot ..."
uv run "${PLOT_HELPER}" \
  --plotfiles "${PLOTFILES[0]}" "${PLOTFILES[1]}" "${PLOTFILES[2]}" \
  --output "${OUTPUT_PLOT}"

echo "Done."
echo "Plot: ${OUTPUT_PLOT}"
