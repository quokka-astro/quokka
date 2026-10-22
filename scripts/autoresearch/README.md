# Kernel autoresearch for Quokka on moth

This is an autoresearch-style loop for one Quokka GPU kernel. An agent edits
Quokka source in a fixed set of paths, and `ar eval` builds each commit on
moth's MI210 (gfx90a) with ROCm's HIP compiler. It then checks the problem's
own test and times the chosen kernel against the current best, keeping or
discarding the commit. `program.md` holds the agent's instructions.

It is adapted from the Loom chemistry-kernel harness in
hrx-system (`experimental/loom-codegen-comparison/autoresearch/`). It has the
same campaign mechanics, but here the toolchain is hipcc and the timing
comes from rocprofv3.

| File | Runs on | Purpose |
| --- | --- | --- |
| `ar` | Mac | Campaign CLI: init, set, setup, kernels, eval, resume, status, deadline, target |
| `remote.sh` | moth | Seed build, overlay of the edited paths, incremental build, test run, paired timing |
| `analyze.py` | moth | rocprofv3 kernel metrics, the kernel list, and the paired-speedup verdict |
| `program.md` | agent | Rules, loop, evidence |

## A campaign, end to end

A campaign is one problem executable, one kernel regex and one set of
editable paths. Run these from a checkout that has this directory:

```sh
A=$PWD/scripts/autoresearch/ar
$A init hydro1 --problem HydroBlast3D --dim 3 \
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

`ar set KEY=VALUE` changes `PROBLEM DIM KERNEL_RE EDIT_PATHS CHECK_ARGS
TIME_ARGS CMAKE_ARGS RUN_PREFIX` until the baseline is recorded. Changing
`PROBLEM`, `DIM` or `CMAKE_ARGS` needs another `ar setup`. `RUN_PREFIX` goes
in front of rocprofv3, for example `mpirun -np 1` if singleton MPI fails.

State lives in `.notes/autoresearch/<TAG>/` in the checkout that holds `ar`
(`results.tsv`, `notes.md`, `campaign.env`, `best.env`, `kernels.txt`,
`runs/<RUN>/`); `ar init` adds `/.notes/` to `.git/info/exclude`. Discarded
commits stay reachable as tags `ar/<TAG>/<RUN>-<status>`. On moth,
everything is under `/mnt/data/bwibking/ar-quokka/<TAG>/`.

## What an eval does

1. **Ships the edited paths.** `git archive` of `EDIT_PATHS` at HEAD goes
   to moth. The commit may change nothing else.
2. **Overlays them.** moth keeps one source tree (the seed with its
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

Both `ar eval` and `ar setup` take moth's shared GPU lock
(`/mnt/data/bwibking/ar/gpu.lock`, the same one the Loom chemistry harness
uses). They also wait up to `--gpu-wait` seconds for other users' GPU
processes to finish.

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

- One plain `ssh moth` connection per command. moth's firewall punishes
  bursts.
- Nothing is compiled on the Mac.
- Agents never estimate the clock: `ar` prints it.
- Static stats are hints, not vetoes.
