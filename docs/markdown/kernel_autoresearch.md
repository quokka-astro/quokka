# Kernel autoresearch

The kernel autoresearch harness in `scripts/autoresearch/` lets an AI coding agent work on making one Quokka GPU kernel faster without supervision. The agent edits the source, and the harness builds, tests and times every change on a real GPU. It keeps a change only if the problem's test still passes and the kernel is measurably faster.

This page explains how the harness works and how to point it at a kernel. `scripts/autoresearch/README.md` has the site-specific setup details. `scripts/autoresearch/program.md` holds the agent's instructions; it is worth reading too, since it is what the agent will follow.

## Prerequisites

You should:

-   Know that each ``amrex::ParallelFor`` launches a GPU kernel (see [Performance tips](performance.md)).
-   Have an account on one of the supported *sites*, which have gfx90a GPUs and ROCm 10.1:
    -   **moth**: an AMD Instinct MI210 workstation. The GPU is shared with other users, so the harness waits for it to be idle and takes a shared lock.
    -   **Setonix** (Pawsey): one MI250X GCD in an interactive `gpu-dev` job that you start yourself. ROCm 10.1 and a recent CMake are installed with pip into a venv once; see the README.
-   Have passwordless ssh to the site from the machine that runs the agent. That machine can run macOS or Linux, and it never compiles Quokka.
-   Have an agent that can run shell commands in a git checkout, such as Claude Code.

## How it works

A *campaign* optimizes one kernel in one problem. It is defined by:

-   a **problem**: the CMake target, for example `HydroBlast3D`. Its test (the exit code of the problem run with its usual inputs) decides correctness;
-   a **kernel regex**: a Python regular expression matched against the demangled kernel names that `rocprofv3` reports. The *kernel time* is the summed GPU time of every launch that matches;
-   the **editable paths**: the only files a commit may change, for example `src/hydro`. They may not include `extern/`, `inputs/` or the problem's own directory, so the agent cannot change the test that judges it.

`ar init` creates a separate git worktree on a branch `autoresearch/<TAG>`. The agent works there in a loop:

1.  Edit the editable paths and commit.
2.  Run `ar eval "one-line description"`, which measures `HEAD` on the site:
    -   it copies the editable paths over the site's copy of the seed tree and rebuilds the problem incrementally;
    -   if the executable is byte-identical to the current best, the verdict is `identical`;
    -   it runs the problem's test once; a nonzero exit is `incorrect`;
    -   it times interleaved pairs of runs, candidate against best, each under `rocprofv3 --kernel-trace`. Each pair gives a speedup of best kernel time over candidate kernel time. Interleaving the pairs cancels slow drifts in clock speed and temperature.
3.  Read the verdict. `ar eval` acts on it:
    -   **keep**: the median speedup reaches the campaign's bar, every pair agrees (one may disagree once there are four or more), and the total GPU time of all kernels did not get more than 1% worse. The commit becomes the new best.
    -   **discard**, **identical**, **incorrect**, **build-fail**, **hang**, **launch-mismatch**: the commit is saved as a tag `ar/<TAG>/<RUN>-<status>` and the worktree is reset to the best commit.
    -   **gpu-busy**, **harness-fail**: nothing changes, and the same commit can be evaluated again later.

The first eval, at the unmodified seed commit, is the *baseline*. It times the seed against itself to measure the noise, and sets the bar to twice the median noise, but never below a 2% speedup.

`launch-mismatch` means the candidate launched the matched kernels a different number of times than the best, usually because the run took a different number of timesteps. Times of runs that did different work are not comparable. The agent cannot keep a change that splits, fuses or renames the kernel; it is told to write such ideas up for a human instead.

## Choosing a kernel

Pick a kernel that matters and a problem whose test would catch a wrong result:

1.  **Find where the time goes.** Profile a representative run, or use the harness: after `ar setup`, `ar kernels` lists the seed's kernels by total GPU time.
2.  **Pick a problem that exercises the kernel.** The problem's test is the only correctness check, so it should compare against a reference solution, not just run to completion. Quokka tests signal failure with a nonzero exit code. Read the test before you start a campaign.
3.  **Size the timing run.** `--check-args` sets the arguments of the correctness run, and `--time-args` sets those of the timing runs (by default the same). Each eval times at least ten runs, so cap a long test with, for example, `max_timesteps=50` in the timing arguments. The kernel should still run long enough to measure, ideally tens of milliseconds or more per run.
4.  **Write the regex.** AMReX `ParallelFor` lambdas appear in the trace as `amrex::launch_global<...>` with the enclosing function in the name, so match on the enclosing function, for example `HydroSystem<.*>::ComputeFluxes`. Check with `ar kernels` that the regex matches only what you mean.
5.  **Choose the editable paths.** Include the directories that hold the kernel and its helpers, and no more. Every extra path is more room for the agent to make changes that aren't about this kernel. Editing a widely included header makes every eval rebuild more of the code, which slows the loop.

## Running a campaign

Run these from a Quokka checkout that has `scripts/autoresearch/`:

```bash
A=$PWD/scripts/autoresearch/ar
$A init hydro1 --site moth --problem HydroBlast3D --dim 3 \
   --edit "src/hydro src/math" \
   --time-args "../inputs/HydroBlast3D.toml max_timesteps=50"
cd ../quokka-ar-hydro1        # the campaign worktree
$A setup                      # ship the seed to the site, build it, profile it once
$A kernels                    # the seed's kernels by total GPU time
$A set KERNEL_RE='HydroSystem<.*>::ComputeFluxes'
$A deadline set-in 480        # optional: stop after 8 hours
$A target 1.3                 # optional: stop once a kept run is 1.3x the baseline
$A eval "baseline"            # the seed against itself: noise and the bar
claude "Read $(dirname "$A")/program.md and run the campaign."
```

The arguments in `--check-args` and `--time-args` are relative to a work directory one level below the source root, like the `tests/` directory where `ctest` runs. `ar set` can change the settings until the baseline is recorded; after that they are frozen, so that every result in the campaign is comparable.

On Setonix, pass `--site setonix` and start an interactive job before `ar setup` and before every session of evals:

```bash
salloc --no-shell -p gpu-dev -A <project>-gpu --gpus-per-node=1 -t 4:00:00
```

`gpu-dev` jobs last at most four hours. The harness refuses to start an eval when the job has less than 30 minutes left (set `AR_MIN_JOB_MINUTES` to change that), and the agent stops when that happens. Start a new job and restart the agent to continue the campaign.

## Following a campaign

The campaign's state lives in the checkout that holds `ar`, under `.notes/autoresearch/<TAG>/`, which git ignores:

-   `results.tsv`: one row per eval, with its verdict, speedup, kernel times and registers;
-   `notes.md`: the agent's lab notebook, with hypotheses, dead ends and a final summary;
-   `runs/<RUN>/`: everything one eval produced, including the build log, the test output and the timing pairs.

`ar status` prints the results table, the best run, the bar, and the time left before the deadline. The agent stops at the deadline or the target; you can also simply stop it.

## Using the results

The campaign branch `autoresearch/<TAG>` holds only kept commits, so its log is the list of changes that made the kernel faster, each with its measured speedup in `results.tsv`. Before upstreaming:

-   Review every commit as you would any pull request. The harness shows that the problem's test passes and that the kernel got faster on one GPU. It does not show that the change is correct for other problems, other dimensions or other GPUs.
-   Rebase the useful commits onto `development` and run the full test suite, including the GPU tests.
-   Discarded experiments stay available as tags `ar/<TAG>/<RUN>-<status>`, in case an idea is worth revisiting by hand.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AR_TAG` | Campaign to act on when not inside its worktree. |
| `AR_WORKTREE` | Where `ar init` creates the worktree (default `../quokka-ar-<TAG>`). |
| `AR_HOST`, `AR_REMOTE_ROOT` | Override the site's ssh host and remote campaign directory. |
| `AR_JOBID` | On Setonix, the job to use when several are running. |
| `AR_MIN_JOB_MINUTES` | On Setonix, the least time a job must have left for an eval to start (default 30). |
