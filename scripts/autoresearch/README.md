# Kernel autoresearch for Quokka GPU kernels

This is an autoresearch-style loop for one Quokka GPU kernel. An agent edits
Quokka source in a fixed set of paths, and `ar eval` builds each commit on a
remote gfx90a GPU with ROCm's HIP compiler. The GPU is moth's MI210 or one
MI250X GCD in an interactive Setonix gpu-dev job (see [Sites](#sites)). It then checks the problem's
own test and times the chosen kernel against the current best, keeping or
discarding the commit. `program.md` holds the agent's instructions.

It is adapted from the Loom chemistry-kernel harness in
hrx-system (`experimental/loom-codegen-comparison/autoresearch/`). It has the
same campaign mechanics, but here the toolchain is hipcc and the timing
comes from rocprofv3.

| File | Runs on | Purpose |
| --- | --- | --- |
| `ar` | local (macOS or Linux) | Campaign CLI: init, set, setup, kernels, eval, resume, status, deadline, target |
| `remote.sh` | remote | Seed build, overlay of the edited paths, incremental build, test run, paired timing |
| `sites/*.sh` | remote | Per-machine toolchain, pinning, GPU lock and job handling (shipped as `bin/site.sh`) |
| `analyze.py` | remote | rocprofv3 kernel metrics, the kernel list, and the paired-speedup verdict |
| `program.md` | agent | Rules, loop, evidence |

## A campaign, end to end

A campaign is one problem executable, one kernel regex and one set of
editable paths. Run these from a checkout that has this directory:

```sh
A=$PWD/scripts/autoresearch/ar
$A init hydro1 --problem HydroBlast3D --dim 3 --site moth \
   --edit "src/hydro src/math" \
   --time-args "../inputs/HydroBlast3D.toml max_timesteps=50"
cd ../quokka-ar-hydro1            # the campaign worktree, branch autoresearch/hydro1
$A setup                          # pack seed + submodules, configure, full HIP build, profile
$A kernels                        # the seed's kernels by total GPU time
$A set KERNEL_RE='HydroSystem<.*>::ComputeFluxes'
$A deadline set-in 240            # optional
$A target 1.3                     # optional: stop at 1.3x the baseline
$A eval "baseline"                # seed vs itself: noise floor and acceptance bar
claude "Read $A/../program.md and run the campaign."
```

`init` options:

- `--problem NAME`: the CMake target, and the executable's name.
- `--dim N`: `AMReX_SPACEDIM`. The default is 3.
- `--site NAME`: where to build and time, `moth` (the default) or
  `setonix`. Each is a file in `sites/`.
- `--edit "PATHS"`: the paths, relative to the repo root, that commits may
  change. They must not include `extern/`, `inputs/` or the problem's own
  `src/problems/NAME`, because the problem's test and inputs judge
  correctness.
- `--kernel REGEX`: a Python `re.search` pattern on rocprofv3's demangled
  kernel names. AMReX `ParallelFor` lambdas show up as
  `amrex::launch_global<...Enclosing::function(...)::{lambda...}>`, so match
  on the enclosing function. The pattern can be left empty until after setup
  and set with `ar set`.
- `--check-args "ARGS"`: the correctness run's arguments. They are shell
  words, so quotes work, and paths are relative to a work directory one
  level below the source root, like ctest's `tests/`. The default is
  `../inputs/NAME.toml`, the same test ctest runs.
- `--time-args "ARGS"`: the timing runs' arguments. The default is the
  check args. Use a shorter or larger setup if the test's run is badly
  sized for timing; for example, cap `max_timesteps`.

`ar set KEY=VALUE` changes `SITE PROBLEM DIM KERNEL_RE EDIT_PATHS CHECK_ARGS
TIME_ARGS CMAKE_ARGS RUN_PREFIX` until the baseline is recorded. Changing
`SITE`, `PROBLEM`, `DIM` or `CMAKE_ARGS` needs another `ar setup`. `RUN_PREFIX` goes
in front of rocprofv3, for example `mpirun -np 1` if singleton MPI fails.

State lives in `.notes/autoresearch/<TAG>/` in the checkout that holds `ar`
(`results.tsv`, `notes.md`, `campaign.env`, `best.env`, `kernels.txt`,
`runs/<RUN>/`); `ar init` adds `/.notes/` to `.git/info/exclude`. Discarded
commits stay reachable as tags `ar/<TAG>/<RUN>-<status>`. Remotely,
everything is under `<root>/<TAG>/`: `/mnt/data/bwibking/ar-quokka` on moth,
`$MYSCRATCH/ar-quokka` on Setonix (`AR_REMOTE_ROOT` overrides it, and
`AR_HOST` the ssh host).

## What an eval does

1. **Ships the edited paths.** `git archive` of `EDIT_PATHS` at HEAD goes
   to the remote side. The commit may change nothing else.
2. **Overlays them.** The remote side keeps one source tree (the seed with its
   submodules) and one build tree per campaign. The edited paths are
   rsynced over the tree in checksum mode, so only changed files get new
   mtimes. Then `cmake --build --target PROBLEM` rebuilds incrementally.
3. **Skips identical code.** If the executable is byte-identical to the
   best's, the verdict is `identical`.
4. **Checks correctness.** It runs the problem once with `CHECK_ARGS`. A
   nonzero exit, which Quokka tests use for failure, means `incorrect`; a
   run past 30 minutes is `hang`. The matched kernel must launch at least
   once.
5. **Times the kernel.** It runs interleaved pairs of candidate and best
   executables with `TIME_ARGS`, each under `rocprofv3 --kernel-trace`.
   The metric is the summed GPU time of every launch matching `KERNEL_RE`.
   Each pair gives speedup = best ms / candidate ms.
6. **Gives a verdict.**
   - `keep`: the median speedup is at least the bar, and every pair agrees,
     except that one may disagree when there are four or more. The total
     time of all kernels must not regress by more than 1%.
   - `launch-mismatch`: the candidate launches the matched kernels a
     different number of times. That usually means a different number of
     timesteps, so the times aren't comparable.
   - `discard`: anything else.

Both `ar eval` and `ar setup` take the site's GPU lock. On moth that is
`/mnt/data/bwibking/ar/gpu.lock`, the same one the Loom chemistry harness
uses, and they also wait up to `--gpu-wait` seconds for other users' GPU
processes to finish. On Setonix the lock is a file in the node's `/tmp`; it
only matters when two campaigns share one job. There, amd-smi sees only the
job's GCD, so the clock telemetry and the process check cover just that GCD.

If the job doing an eval dies (a dropped node, or a Setonix job reaching its
time limit), the run is recorded as `harness-fail reason=job-died`, which
changes nothing; re-run `ar eval` on the same commit.

## Sites

A site file in `sites/` defines `SITE`, `GPU_LOCK`, `ROCPROF`, `AMDSMI`
(empty: no foreign-process check and no clock telemetry), `PYTHON`,
`BUILD_PREFIX`, `BUILD_JOBS`, `GPU_PREFIX` and `CONFIGURE_ARGS` (compilers,
`CMAKE_PREFIX_PATH`), and three functions:

- `site_enter ARGS...`: on a batch system, re-run `remote.sh ARGS...` on the
  GPU job's node (`exec ssh`); returns 1 when there is no usable job.
- `site_check`: before setup or an eval starts, returns 1 if the GPU will go
  away mid-run.
- `site_env`: the build and run environment; it may set the variables above.

### moth

ssh lands on the GPU host, and nothing gates access to the GPU, so the shared
lock and the wait for other users' processes keep timing runs alone.

### Setonix

Only an interactive job can use a GPU, so the remote side runs on the job's
node. ar's ssh lands on a login node, `remote.sh` finds your running gpu-dev
job with `squeue` and re-runs itself there over ssh, and from then on works
as on moth.

1. **Install ROCm 10.1 once**, from AMD's pip wheels, into the venv that
   `sites/setonix.sh` expects. The system's ROCm modules are older. Keep
   pip's cache off `$HOME`, whose quota is small:

   ```sh
   module load cray-python/3.11.7
   export PYTHONNOUSERSITE=1 PIP_CACHE_DIR=$MYSCRATCH/pip-cache TMPDIR=$MYSCRATCH/tmp
   python3 -m venv $MYSOFTWARE/ar-quokka/venv-rocm-10.1
   V=$MYSOFTWARE/ar-quokka/venv-rocm-10.1/bin
   $V/python -s -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
     "rocm[libraries,devel,device-gfx90a]==10.1.0" "cmake>=3.31" ninja
   $V/rocm-sdk init        # expands the devel tree: headers, CMake packages
   ```

   `PYTHONNOUSERSITE` and `-s` keep packages in `~/.local` out of the venv.
   Quokka needs CMake 3.31 or later (policy CMP0179); with the 3.30 module
   the HIP device link sees libamrex twice and fails on duplicate symbols.
2. **Start a job** before `ar setup` or `ar eval`. It can be an interactive
   shell (`salloc` in tmux) or a bare allocation:

   ```sh
   salloc --no-shell -p gpu-dev -A pawsey0807-gpu --gpus-per-node=1 -t 4:00:00
   ```

   With one GCD the job has 8 cores, so a full build of a problem takes a
   while. If you have several running gpu-dev jobs, set `AR_JOBID`.
3. **Mind the clock.** gpu-dev jobs last at most 4 hours. `remote.sh`
   refuses to start setup or an eval when the job has less than
   `AR_MIN_JOB_MINUTES` (default 30) left; that eval is recorded as
   `harness-fail reason=gpu-job`. A new job may land on another node; the
   source and build trees are on `/scratch`, so the next eval simply
   continues there, and every eval re-times the best on the same node.

MPI and HDF5 come from the modules of
`scripts/hpc_profiles/setonix-gpu.profile` (cray-mpich, without the GPU
transport layer, since one rank never sends device buffers).

## Requirements on moth

These were in place on 2026-10-22, though they haven't been exercised by
this harness yet:
- ROCm 10.1 at `/opt/rocm/core-10.1`, with `rocprofv3`, `amd-smi` and
  `llvm/bin/clang++`;
- system HDF5 (`/usr/include/hdf5.h`, `/usr/lib64/libhdf5*`), which Quokka
  requires;
- OpenMPI at `/usr/lib64/openmpi/bin`, since Quokka forces `AMReX_MPI=ON`.
  remote.sh puts it on `PATH`. MPICH is also at `/usr/lib64/mpich`;
- `/mnt/data/bwibking/venv-py312` (cmake, ninja, Python 3.12), plus
  `rsync`.

The first `ar setup` is the validation step. The likely problems:
- **configure:** HIP detection with ROCm 10.1's clang, or MPI. Add flags
  through `ar set CMAKE_ARGS=...`.
- **singleton MPI start-up:** try `RUN_PREFIX='mpirun -np 1'`.
- **rocprofv3 CSV column names:** analyze.py looks columns up tolerantly,
  but check one `kernels.txt`.

On gfx90a, rocprofv3's VGPR column is wrong for kernels whose
`ACCUM_OFFSET` is above 128 (ROCm/rocm-systems#13686). Treat `vgpr` as a
hint.

## Rules carried over from the Loom harness

- One plain ssh connection per command. moth's firewall punishes bursts.
- Nothing is compiled on the local machine.
- Agents never estimate the clock: `ar` prints it.
- Static stats are hints, not vetoes.
