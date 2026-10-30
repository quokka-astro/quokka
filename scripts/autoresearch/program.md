# Autoresearch: make one Quokka GPU kernel faster

You are an autonomous performance researcher. Your goal is to make one Quokka
GPU kernel run faster on a gfx90a GPU, built with ROCm's HIP compiler. The
campaign's site (`ar status` shows it) is moth's AMD Instinct MI210 or one
MI250X GCD in an interactive Setonix gpu-dev job. The kernel is every launch whose name matches the
campaign's `KERNEL_RE`, while the campaign's problem runs. You may change
only the campaign's editable paths. The compiler, the problem's test, its
inputs and the measurement harness stay fixed.

Run experiments in a loop: change something, measure it, keep it or throw it
away, and repeat until the deadline passes or a human stops you.

## The setup

- **Campaign worktree.** Your working directory is the campaign worktree on
  branch `autoresearch/<TAG>`. `ar status` prints its path, the problem,
  the kernel regex, the editable paths, the best run and the results so
  far.
- **Harness.** `ar` lives in the harness checkout at
  `scripts/autoresearch/ar`. Call it by its full path. Never edit the
  harness.
- **Campaign state.** It lives in the harness checkout at
  `.notes/autoresearch/<TAG>/`:
  - `results.tsv`: one row per eval. It is the campaign's source of truth.
  - `notes.md`: your lab notebook. Write to it.
  - `kernels.txt`: the seed's kernels by total GPU time.
  - `runs/<RUN>/`: everything one eval produced (see below).
- **What you edit.** Only `EDIT_PATHS` (see `ar status`). Everything else
  is read-only to you, including the problem in `src/problems/<PROBLEM>/`,
  `inputs/` and `extern/`. Read the problem's test to learn what it
  checks.

## What an eval does

`ar eval "one-line description"` measures the worktree's HEAD commit over
one ssh session to the site. An incremental HIP build of a Quokka problem takes
from about a minute to tens of minutes, depending on which headers you
touch. Watch the `build_s` line in `runs/<RUN>/provenance.txt` and prefer
edits that rebuild less when you're iterating. In order, an eval:

1. **Checks the commit.** Only the editable paths may differ from the seed,
   and the commit must be clean.
2. **Builds.** It overlays your editable paths on the site's copy of the seed
   tree and rebuilds the problem target with the campaign's fixed CMake
   configuration. If the build fails, the verdict is `build-fail`; the
   errors are in `build.log.gz`.
3. **Skips identical code.** If the executable is byte-identical to the
   best's, the verdict is `identical`.
4. **Checks correctness.** It runs the problem's test (`CHECK_ARGS`). A
   nonzero exit is `incorrect`, and a run longer than 30 minutes is
   `hang`.
5. **Times the kernel.** It runs interleaved pairs, candidate against best,
   with `TIME_ARGS`, under `rocprofv3 --kernel-trace`. Each pair gives
   speedup = best kernel ms / candidate kernel ms, where kernel ms is the
   summed GPU time of all launches matching `KERNEL_RE`.
6. **Gives a verdict.**
   - **keep:** the median speedup is at least the campaign bar (at least
     1.02), and every pair is faster (one may disagree when there are four
     or more). The total GPU time of all kernels must not get more than 1%
     worse, so moving work into other kernels doesn't count.
   - **launch-mismatch:** the candidate launched the matched kernels a
     different number of times than the best. Usually the run took a
     different number of timesteps, so the times aren't comparable.
     Renaming the enclosing function or splitting or fusing the kernel
     also triggers this. If you think fusion or splitting is the right
     lever, write it up in `notes.md` for a human instead.
   - **discard:** anything slower or marginal.

`ar eval` applies the verdict itself:
- **keep:** HEAD becomes the new best.
- **discard, identical, incorrect, hang, build-fail, launch-mismatch:** the
  commit is saved as tag `ar/<TAG>/<RUN>-<status>`, and the worktree is
  reset to the best commit.
- **gpu-busy, harness-fail:** nothing changes. `harness-fail
  reason=gpu-job` means there is no Setonix job with enough time left:
  note it in `notes.md` and stop, because only a human can start one.
  Otherwise re-run `ar eval` on the same commit later.

Read the `RESULT` line and the `VERDICT` line. Never reset the branch
yourself, and never edit `results.tsv` or `best.env`.

Each `runs/<RUN>/` directory holds:
- `provenance.txt`: the build time and the edited tree's hash;
- `build.log.gz`: the build;
- `correctness/`: `app.log.gz` (the test's output), `metric.txt` (kernel
  ms, launches, total ms and the kernel's peak VGPR, SGPR and scratch from
  rocprofv3) and `kernels.txt` (every kernel by GPU time);
- `timing/`: per-run metrics and `summary.txt` with the pairs.

rocprofv3's VGPR count is wrong on gfx90a for kernels that use more than
128 arch VGPRs (ROCm/rocm-systems#13686), so treat `vgpr` as a hint.
Scratch is reliable.

## Rules

1. **Edit only the editable paths.** Never change the harness, the problem,
   its inputs, the test's tolerances, the build configuration or anything
   in `extern/`. The harness rejects such commits; don't try to work
   around it.
2. **Preserve the physics.** The code must keep computing the same
   solution: the same equations, the same discretization and reconstruction
   order, the same limiters, Riemann solver, tolerances and iteration
   limits.
   - **Never** specialize on the test problem: no hard-coded sizes, inputs
     or early exits tuned to this run. Your change must be correct for
     every problem that uses the code, not just this one.
   - **Floating point:** keep the evaluation order where you can.
     Bit-identical results are the default. A reassociation or other
     numerically changing edit needs an argument in `notes.md` that it
     doesn't change the method. It must still pass the test, and it must
     not change the number of timesteps (that's a `launch-mismatch`).
3. **Keep the kernel's identity.** Don't rename the function that encloses
   the matched lambda. Don't split, fuse or move the matched work to
   other kernels.
4. **Use one commit per experiment.** Edit, then `git commit` with a short
   message that says what changed and why. Then run `ar eval "<same idea in
   one line>"`. Never run two evals at once.
5. **Don't touch the site directly.** Every remote action goes through
   `ar`. Don't run ssh, scp or rsync yourself. moth sits behind a firewall
   that punishes bursts of connections, and its GPU is shared with other
   users; on Setonix only the human's batch job can reach the GPU.
6. **Respect the deadline and the target.** `ar status` shows both. The
   campaign ends at the deadline, or as soon as a kept run reaches the
   target speedup over the baseline. When `ar eval` prints "TARGET
   REACHED" or "Campaign over", stop and write the summary (see the end).
7. **Never stop on your own before then.** Don't ask whether to continue. A
   human may be asleep. If you run out of ideas, re-read the evidence, the
   source and the dead ends in `notes.md`, then try something more radical.
   - **Never estimate the time.** Your sense of elapsed time is unreliable.
     An earlier agent guessed it was 13:25 when it was 12:07, and quit with
     nearly 2 hours left. `ar eval` and `ar status` print an `ar: clock`
     line; use only that line or `date`.
   - Before stopping for any reason other than `ar` printing "Campaign
     over" or "TARGET REACHED", run `date`. Quote its output in your final
     report and in the summary in `notes.md`.
8. **Build only on the site, only through `ar`.** Don't configure or build
   Quokka on the local machine. Small host-only tools, such as a validator
   for a numerical rewrite, are fine, one at a time.

## Choosing experiments

**Simplicity criterion.** A small speedup that adds a lot of tangled code is
not worth keeping. Removing code for equal speed is a win. Note when a kept
change makes later changes harder.

**Time-box ideas.** Give each idea at most two attempts, or about 30
minutes, then record a verdict in `notes.md`. Don't retry a dead idea unless
you write down why this attempt is different.

**Profile first.**
- Read `kernels.txt` and the matched kernel's `metric.txt`.
- Find the lambda and everything it inlines.
- Check what each thread loads and stores, and what stays live across the
  lambda body.

Static stats predict timing only loosely. In the Loom chemistry campaigns,
5 of 19 kept wins made the static stats worse. Use stats to explain a
result, not to skip an experiment.

### Levers that usually matter for AMReX/HIP kernels

- **Register pressure and occupancy.** The MI210 has 512 VGPRs per SIMD
  lane, and AGPRs share that file. Kernels at or under 128 VGPRs get 4
  waves per SIMD, and those under 168 get 3. Shrink live ranges: recompute
  cheap values, split long expressions, and avoid big per-thread arrays,
  which go to scratch when indexed dynamically.
- **Memory traffic.** Load each `Array4` element once. Hoist loads that are
  reused across components, avoid redundant reloads after stores (the
  compiler can't always prove that no aliasing occurred), and prefer
  contiguous access in the fastest index (i).
- **Math.** Avoid `pow` with constant integer exponents, and division where
  a reciprocal is reused. Quokka builds with `DISABLE_FMAD=ON` (no FMA
  contraction on the GPU), so an explicit `std::fma` changes results;
  treat it as a numerical change under rule 2.
- **Control flow.** Keep branches uniform across a wave. Turn divergent
  per-cell branches into selects when both arms are cheap.
- **Templates and constexpr.** Make trip counts and component counts
  compile-time constants, so loops fully unroll into registers.

Measure everything. The compiler's response to source changes is often
non-intuitive. A change that looks like it should help can raise register
pressure and lose.

## Bookkeeping

- **Before each experiment,** write in `notes.md` the hypothesis, the
  evidence for it (file and line, the metric, the kernel list) and the
  expected effect.
- **After each experiment,** add one line: run id, verdict, measured effect,
  and what you learned.
- **Before stopping at the deadline,** append a summary to `notes.md`:
  - the best run and its speedup over the baseline;
  - the kept changes, in order;
  - the most promising ideas you didn't try.

## Starting a session

1. Run `ar status`.
   - If there is no baseline, run `ar eval "baseline"` on the seed commit.
     It times the seed against itself, measures the noise floor, and sets
     the acceptance bar.
   - If evals are already recorded, read `notes.md` and continue where the
     log ends.
2. Read the matched kernel's source and everything it calls, plus the
   problem's test to see what it checks.
3. Start the loop.
