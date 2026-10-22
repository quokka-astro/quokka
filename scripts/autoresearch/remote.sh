#!/usr/bin/env bash
# Remote side of the Quokka kernel autoresearch harness. Only `ar` calls this,
# over one ssh session per command; see README.md. bin/site.sh (a copy of
# sites/SITE.sh) supplies the machine: toolchain, CPU pinning, the GPU lock,
# and on batch systems the hop from the login node to the job's node.
#
# SETTINGS is the campaign's settings as eight words:
#   PROBLEM DIM KERNEL_RE EDIT_PATHS CHECK_ARGS TIME_ARGS CMAKE_ARGS RUN_PREFIX
#
#   remote.sh setup-start TAG SEED SETTINGS   (bin/ and src.tar.gz were unpacked
#                                            into the campaign dir)
#   remote.sh eval-start TAG RUN BEST PAIRS BAR CONTENDED GPU_WAIT SETTINGS
#                                            (stdin: tar.gz of EDIT_PATHS;
#                                            stdout: bundle)
#   remote.sh resume TAG RUN                 (stdout: bundle)
#
# Long jobs run detached (setsid nohup) and log to jobs/; the ssh session only
# follows the log, so a dropped connection loses nothing and `ar resume`
# reattaches. Progress goes to stderr; stdout carries only data.
set -uo pipefail
# This script lives in the campaign directory's bin/.
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TARGET=gfx90a
BUILD_TIMEOUT=7200
CHECK_TIMEOUT=1800
TIME_TIMEOUT=900
# shellcheck source=sites/moth.sh
source "$D/bin/site.sh"

cmd=${1:?command}; tag=${2:?tag}; shift 2
[[ $tag =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad tag: $tag" >&2; exit 2; }
[[ $(basename "$D") == "$tag" ]] || { echo "remote.sh is in $D, not in campaign $tag" >&2; exit 2; }

log() { echo "[$(date +%H:%M:%S)] $*"; }

settings() {  # settings WORDS... : the eight campaign settings
  PROBLEM=$1 DIM=$2 KERNEL_RE=$3 EDIT_PATHS=$4 CHECK_ARGS=$5 TIME_ARGS=$6 CMAKE_ARGS=$7 RUN_PREFIX=$8
}

detach() {  # detach NAME ARGS... : run `remote.sh ARGS...` detached
  mkdir -p "$D/jobs"
  setsid nohup bash "$D/bin/remote.sh" "${@:2}" > "$D/jobs/$1.log" 2>&1 < /dev/null &
  echo "$! $(hostname)" > "$D/jobs/$1.pid"
}

follow() {  # follow NAME : stream the job log to stderr until the job exits
  local pid host
  read -r pid host < "$D/jobs/$1.pid" 2>/dev/null || { echo "no job $1" >&2; return 1; }
  if [[ $host == "$(hostname)" ]]; then
    tail -n +1 -f --pid="$pid" "$D/jobs/$1.log" >&2
  else
    # A batch job that ended took the detached job with it; a new one may be
    # on another node.
    echo "job $1 ran on $host, not $(hostname); it ended with its batch job" >&2
    cat "$D/jobs/$1.log" >&2
  fi
}

salvage() {  # salvage RUN : a result for a run whose job died before finishing
  local out=$D/runs/$1
  [[ -f $out/bundle.tgz ]] && return 0
  mkdir -p "$out"
  echo "RESULT status=harness-fail reason=job-died" > "$out/result.txt"
  (cd "$out" && tar czf bundle.tgz result.txt)
}

early_result() {  # early_result STATUS KEY=VALUE... : a bundle for an eval that never started
  local tmp
  tmp=$(mktemp -d)
  echo "RESULT status=$*" | tee "$tmp/result.txt" >&2
  (cd "$tmp" && tar czf - result.txt)
  rm -rf "$tmp"
  exit 0
}

foreign_gpu_procs() {  # GPU processes; called only while none of ours runs
  [[ -n $AMDSMI ]] || return 0
  "$AMDSMI" process 2>/dev/null | awk '/NAME:/{n=$2} /PID:/{print $2 " " n}' || true
}

build_key() { echo "problem=$PROBLEM dim=$DIM cmake_args=$CMAKE_ARGS"; }

build_problem() {  # build_problem LOG : incremental build of the problem target
  "${BUILD_PREFIX[@]}" timeout $BUILD_TIMEOUT cmake --build "$D/build" --target "$PROBLEM" -j "$BUILD_JOBS" > "$1" 2>&1
}

problem_exe() { find "$D/build" -type f -name "$PROBLEM" -perm -u+x | head -1; }

gpu_acquire() {  # gpu_acquire GPU_WAIT CONTENDED : take the lock, wait out other users
  exec 9> "$GPU_LOCK"
  flock 9
  local waited=0 f
  while f=$(foreign_gpu_procs); [[ -n $f ]]; do
    if (( waited >= $1 )); then
      if [[ $2 == 1 ]]; then log "GPU still shared after ${waited}s; timing anyway (contended runs are flagged)"; return 0; fi
      log "GPU busy with: $f"; return 1
    fi
    (( waited == 0 )) && log "GPU in use ($f); waiting up to $1s"
    sleep 30; waited=$((waited + 30))
  done
}

# run_profiled EXE ARGS DIR TIMEOUT : run EXE once under rocprofv3's kernel
# trace, in a scratch directory one level below the source root (so the
# problem's relative paths resolve as they do under ctest). Leaves DIR/app.log,
# DIR/metric.txt and DIR/kernels.txt; returns the app's exit code.
run_profiled() {
  local exe=$1 dir=$3 limit=$4 work rc argv=() prefix=()
  # ARGS and RUN_PREFIX are shell words, so quoted values keep their spaces.
  eval "argv=($2)"
  eval "prefix=($RUN_PREFIX)"
  mkdir -p "$dir"
  work=$(mktemp -d "$D/src/arwork.XXXXXX")
  (cd "$work" && timeout "$limit" "${GPU_PREFIX[@]}" "${prefix[@]}" \
     "$ROCPROF" --kernel-trace --output-format csv -d "$dir/prof" -o trace -- "$exe" "${argv[@]}") > "$dir/app.log" 2>&1
  rc=$?
  rm -rf "$work"
  $PYTHON "$D/bin/analyze.py" metric "$dir/prof" "$KERNEL_RE" > "$dir/metric.txt" 2>&1
  $PYTHON "$D/bin/analyze.py" kernels "$dir/prof" > "$dir/kernels.txt" 2>&1
  rm -rf "$dir/prof"
  return $rc
}

setup_job() {  # setup_job SEED SETTINGS...
  local seed=$1 s=$D/src b=$D/build
  shift; settings "$@"
  log "setup $tag seed=$seed problem=$PROBLEM dim=$DIM"
  rm -rf "$s" "$b" "$D/harness" "$D/setup.ok"
  mkdir -p "$s" "$D/harness" "$D/runs" "$D/inbox"
  tar xzf "$D/src.tar.gz" -C "$s" || { echo "SETUP fail untar"; return 1; }
  rm -f "$D/src.tar.gz"
  log "configure"
  # shellcheck disable=SC2086  # CMAKE_ARGS is a word list
  cmake -G Ninja -S "$s" -B "$b" -DCMAKE_BUILD_TYPE=Release \
    -DAMReX_SPACEDIM="$DIM" -DAMReX_GPU_BACKEND=HIP -DAMReX_AMD_ARCH=$TARGET \
    -DQUOKKA_PYTHON=OFF "${CONFIGURE_ARGS[@]}" \
    $CMAKE_ARGS > "$D/jobs/configure.log" 2>&1 ||
    { tail -40 "$D/jobs/configure.log"; echo "SETUP fail configure"; return 1; }
  log "build $PROBLEM (full build)"
  build_problem "$D/jobs/build.log" ||
    { grep -A12 -E 'FAILED:|error:' "$D/jobs/build.log" | head -60; echo "SETUP fail build"; return 1; }
  local exe
  exe=$(problem_exe)
  [[ -n $exe ]] || { echo "SETUP fail no executable named $PROBLEM in the build tree"; return 1; }
  cp "$exe" "$D/harness/seed-exe"
  build_key > "$D/build.key"
  # One profiled run lists the GPU kernels, so the campaign can pick KERNEL_RE.
  log "profile the seed (TIME_ARGS) to list its kernels"
  gpu_acquire 1800 0 || { echo "SETUP fail gpu-busy"; return 1; }
  run_profiled "$D/harness/seed-exe" "$TIME_ARGS" "$D/harness/seed-profile" $TIME_TIMEOUT
  local rc=$?
  flock -u 9
  [[ $rc == 0 ]] || { tail -30 "$D/harness/seed-profile/app.log"; echo "SETUP fail seed run rc=$rc"; return 1; }
  echo "KERNELS"; cat "$D/harness/seed-profile/kernels.txt"; echo "END KERNELS"
  echo "seed=$seed problem=$PROBLEM dim=$DIM site=$SITE host=$(hostname)" > "$D/setup.ok"
  log "setup done: $(cat "$D/setup.ok")"
  echo "SETUP ok"
}

time_one() {  # time_one IDX LABEL EXE TIMING_DIR
  local dir=$4/run$1-$2 t0 t1 rc f
  f=$(foreign_gpu_procs | wc -l)
  t0=$(date +%s.%N)
  run_profiled "$3" "$TIME_ARGS" "$dir" $TIME_TIMEOUT
  rc=$?
  t1=$(date +%s.%N)
  f=$((f + $(foreign_gpu_procs | wc -l)))
  echo "$1 $2 $rc $t0 $t1 $dir/metric.txt $f" >> "$4/runs.txt"
  log "  run $1 $2 rc=$rc foreign=$f $(cat "$dir/metric.txt")"
}

overlay() {  # overlay STAGE : make the source tree's EDIT_PATHS match STAGE's
  # Checksum mode without times: only changed files get a new mtime, so the
  # build stays incremental and never misses a change.
  local p
  for p in $EDIT_PATHS; do
    if [[ -d $1/$p ]]; then
      mkdir -p "$D/src/$p"
      rsync -rlpc --delete "$1/$p/" "$D/src/$p/" || return 1
    elif [[ -e $1/$p ]]; then
      rsync -lpc "$1/$p" "$D/src/$p" || return 1
    else
      rm -rf "${D:?}/src/$p"
    fi
  done
}

eval_job() {  # eval_job RUN BEST PAIRS BAR CONTENDED GPU_WAIT SETTINGS...
  local run=$1 best=$2 pairs=$3 bar=$4 contended=$5 gpu_wait=$6
  shift 6; settings "$@"
  local out t_start
  t_start=$(date +%s)
  out=$D/runs/$run
  finish() {  # finish STATUS [KEY=VALUE...]
    local stats=""
    [[ -f $out/correctness/metric.txt ]] && stats=$(cat "$out/correctness/metric.txt")
    echo "RESULT status=$1 ${*:2} $stats minutes=$(( ($(date +%s) - t_start + 30) / 60 ))" > "$out/result.txt"
    (cd "$out" || exit
     local f files=()
     for f in result.txt provenance.txt build.log correctness timing; do
       [[ -e $f ]] && files+=("$f")
     done
     tar czf bundle.tgz "${files[@]}")
    log "$(cat "$out/result.txt")"
  }
  mkdir -p "$out/stage"
  log "eval $tag run=$run best=$best pairs=$pairs bar=$bar kernel=/$KERNEL_RE/"
  tar xzf "$D/inbox/$run.tgz" -C "$out/stage" && rm -f "$D/inbox/$run.tgz" || { finish harness-fail reason=untar; return; }
  [[ -f $D/setup.ok ]] || { finish harness-fail reason=no-setup; return; }
  [[ $(build_key) == "$(cat "$D/build.key")" ]] || { finish harness-fail reason=settings-changed-rerun-setup; return; }
  if [[ $best != none && ! -f $D/runs/$best/exe ]]; then finish harness-fail reason=best-missing; return; fi
  overlay "$out/stage" || { finish harness-fail reason=overlay; return; }
  echo "run=$run best=$best $(cat "$D/setup.ok") edit_sha256=$(cd "$out/stage" && find . -type f | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)" > "$out/provenance.txt"
  rm -rf "$out/stage"

  log "build $PROBLEM"
  local c0 c1
  c0=$(date +%s)
  build_problem "$out/build.log"
  local brc=$?
  c1=$(date +%s)
  if [[ $brc != 0 ]]; then
    grep -A12 -E 'FAILED:|error:' "$out/build.log" | head -60
    finish build-fail rc=$brc; return
  fi
  cp "$(problem_exe)" "$out/exe"
  gzip -f "$out/build.log"
  echo "build_s=$((c1 - c0))" >> "$out/provenance.txt"
  if [[ $best != none ]] && cmp -s "$out/exe" "$D/runs/$best/exe"; then
    finish identical to=$best; return
  fi

  # Everything below holds the GPU lock.
  gpu_acquire "$gpu_wait" "$contended" || { finish gpu-busy waited_s="$gpu_wait"; return; }
  log "correctness: $PROBLEM $CHECK_ARGS"
  run_profiled "$out/exe" "$CHECK_ARGS" "$out/correctness" $CHECK_TIMEOUT
  local rc=$?
  tail -15 "$out/correctness/app.log"
  gzip -f "$out/correctness/app.log"
  [[ $rc == 124 ]] && { flock -u 9; finish hang stage=correctness; return; }
  [[ $rc == 0 ]] || { flock -u 9; finish incorrect rc=$rc; return; }
  grep -q 'launches=[1-9]' "$out/correctness/metric.txt" ||
    { flock -u 9; finish harness-fail reason=kernel-not-launched; return; }

  local t=$out/timing ref=$D/runs/$best/exe idx=0 p
  [[ $best == none ]] && ref=$out/exe  # baseline: A/A pairs measure the noise floor
  mkdir -p "$t"; : > "$t/runs.txt"
  local tel=""
  if [[ -n $AMDSMI ]]; then
    PYTHONUNBUFFERED=1 "$AMDSMI" monitor -p -u -t -w 1 --csv > "$t/telemetry.csv" 2>/dev/null &
    tel=$!
  fi
  pair() {  # odd pairs run candidate first, even pairs best first
    if (( $1 % 2 )); then idx=$((idx+1)); time_one $idx cand "$out/exe" "$t"; idx=$((idx+1)); time_one $idx best "$ref" "$t"
    else idx=$((idx+1)); time_one $idx best "$ref" "$t"; idx=$((idx+1)); time_one $idx cand "$out/exe" "$t"; fi
  }
  log "timing: $pairs interleaved pairs against $([[ $best == none ]] && echo "itself (A/A)" || echo "$best")"
  for ((p = 1; p <= pairs; ++p)); do pair $p; done
  sleep 2
  $PYTHON "$D/bin/analyze.py" verdict "$t" "$bar" "$contended" > "$t/summary.txt"
  local extra
  extra=$(sed -nE 's/^FLAGGED_PAIRS ([0-9]+)$/\1/p' "$t/summary.txt")
  if (( ${extra:-0} > 0 )); then
    log "re-timing $extra flagged pair(s)"
    for ((p = pairs + 1; p <= pairs + extra && p <= pairs + 3; ++p)); do pair $p; done
    sleep 2
    $PYTHON "$D/bin/analyze.py" verdict "$t" "$bar" "$contended" > "$t/summary.txt"
  fi
  [[ -n $tel ]] && kill "$tel" 2>/dev/null
  flock -u 9
  cat "$t/summary.txt"
  local res verdict
  res=$(sed -n 's/^RESULT //p' "$t/summary.txt")
  if grep -q 'failed_runs=[1-9]' <<<"$res"; then verdict=incorrect
  elif grep -q 'clean_pairs=[01] ' <<<"$res"; then verdict=gpu-busy
  elif grep -q 'launch_match=0' <<<"$res"; then verdict=launch-mismatch
  elif [[ $best == none ]]; then verdict=baseline
  elif grep -q 'meets_bar=1' <<<"$res"; then verdict=keep
  else verdict=discard; fi
  # shellcheck disable=SC2086  # res is KEY=VALUE words
  finish $verdict $res
}

# On a batch system, site_enter re-runs the command on the GPU job's node.
case $cmd in
  setup-start|eval-start|resume)
    site_enter "$cmd" "$tag" "$@" || case $cmd in
      setup-start) echo "SETUP fail gpu-job"; exit 1 ;;
      eval-start) early_result harness-fail reason=gpu-job ;;
      resume) exit 1 ;;
    esac ;;
esac
site_env
mkdir -p "${TMPDIR:-/tmp}" "$(dirname "$GPU_LOCK")"

case $cmd in
  setup-start)
    site_check || { echo "SETUP fail gpu-job"; exit 1; }
    detach setup setup-job "$tag" "$@"
    follow setup ;;
  setup-job) setup_job "$@"; echo "JOB-EXIT $?" ;;
  eval-start)
    run=$1
    [[ -f $D/setup.ok ]] || early_result harness-fail reason=no-setup
    [[ -e $D/runs/$run ]] && early_result harness-fail reason=run-exists
    site_check || early_result harness-fail reason=gpu-job
    mkdir -p "$D/inbox"; cat > "$D/inbox/$run.tgz"
    detach "eval-$run" eval-job "$tag" "$@"
    follow "eval-$run"
    salvage "$run"
    cat "$D/runs/$run/bundle.tgz" ;;
  eval-job) eval_job "$@"; echo "JOB-EXIT $?" ;;
  resume)
    follow "eval-$1"
    salvage "$1"
    cat "$D/runs/$1/bundle.tgz" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
