# Plan: isolate the performance regression in efc47cf9

**Status: planned only. Do not execute until the user requests it.**

## Objective and scope

Identify the smallest source changes in Quokka commit
`efc47cf90657e1eee74bb4da01e5b3c0f0597c3a` (PR #335, “Compute pressure from
microphysics eos”) that explain the performance loss relative to its parent,
`aa288b57`. Establish both the responsible code changes and the GPU execution
mechanism, with controlled application measurements.

Do not assume the loss comes from arithmetic inside Microphysics. Separate:

- Changes to EOS arithmetic and input validation.
- Changes to Quokka's EOS call sites, data loading, and repeated calculations.
- Changes from compile-time constants to runtime parameters.
- Changes to numerical behavior, including floors and state-validity decisions.
- Compiler effects such as common-subexpression elimination, register pressure,
  occupancy, and generated division/square-root instructions.

Keep the later FOFC/RK2 rewrite (`6b0e0c8e`) out of this experiment. Do not use
current Quokka or synthetic EOS microbenchmarks as substitutes for measuring
these historical revisions.

## Evidence to reproduce

The saved historical comparison used identical dependencies, compiler, and
flags on both sides:

| Revision | Single MI210, Mupdates/s | Full Frontier node, Mupdates/s |
|---|---:|---:|
| Parent aa288b57 | 142.407 | 1130.2 |
| Commit efc47cf9 | 90.615 | 1019.9 |

These are prior observations, not acceptance targets that new runs must match
exactly. The full-node loss was about 9.8% of throughput, or 10.8% additional
time per zone update. The single-GPU loss was much larger; do not multiply
single-GPU throughput by eight or assume the same mechanism dominates both.

References:

- `build/perf-investigation/REPORT.md`
- `build/perf-investigation/compare-eos-node.submit`
- Saved binaries `HydroBlast3D-pre-pressure-eos` and
  `HydroBlast3D-post-pressure-eos` under `build/perf-investigation/`.
- `build/eos-investigation/REPORT.md` documents a later experiment on current
  Quokka. Its large EOS-only speedups and negligible application speedup do not
  establish the cause of this historical regression.

## 1. Freeze and verify the historical experiment

1. Create separate experimental source/build trees under
   `build/eos-commit-isolation/`. Preserve the current checkout, its uncommitted
   Microphysics optimization, all prior artifacts, and `frontier-1node.submit`.
2. Recover the exact configurations and compatibility patches from the saved
   historical builds. Inspect helper scripts before reuse: the generic
   `prepare_historical.py` follows each revision's submodule pins and does not
   by itself reproduce the dependency-frozen comparison.
3. Freeze AMReX at `de7c6189623bc3ba178a380e703e3173f33f78ef` and Microphysics at
   `ced7cc41dbafae11ed587c1d446d80e442b9b967`, as in the measured comparison.
   Freeze fmt and all other dependencies to the actual saved build revisions.
   Do not accidentally incorporate the current Microphysics patch.
4. Apply identical modern-HIP compatibility edits to both sides: gfx90a wave64,
   the HIP pointer-attribute rename, and the historical zero-length scalar-array
   compatibility patch. Save these separately from experimental patches.
5. Record source/dependency hashes, complete compiler/link commands, compiler
   version, environment, input checksum, GPU model, rank placement, and binary
   hashes. Keep optimization flags and floating-point settings identical.
6. Rebuild untouched historical endpoints and reproduce a statistically clear
   gap. If the gap does not reproduce, resolve that before making ablations.

The native Microphysics submodule bump in PR #335 is a separate question. It
was held fixed in the comparison above. If native dependency behavior is later
examined, label it as a separate experiment; do not mix it into source-hunk
attribution. One native historical Microphysics object was unavailable locally
when inspecting the diff, so do not assume it can be checked out without further
preparation.

## 2. Establish measurement and correctness controls

- Use the saved legacy input
  `build/perf-investigation/source-pre-fofc/tests/benchmark_unigrid_512.in` for
  both historical revisions. Initially use 100 timesteps and a 256^3 domain on
  one GPU, matching the earlier experiment. Use the unchanged 512^3 input on a
  full node with eight ranks, seven CPUs per rank, one GPU per rank, closest GPU
  binding, GPU-aware MPI, and `TMPDIR=/tmp`.
- Run only one local GPU experiment at a time. Avoid concurrent compilation
  during throughput measurements. Check for other GPU users; prefer a dedicated
  allocation if shared-node interference prevents stable measurements.
- Use a warmup, then at least three paired measurements with interleaved order
  (A/B, B/A, A/B). Record all samples, median, spread, and time per zone update.
  Increase repetitions only when the observed effect is comparable to noise.
- Confirm equal grids, zone-update counts, step counts, and enabled physics.
  Record timestep histories, conservation diagnostics, and final-state
  differences. A change in numerical work must not masquerade as faster execution.
- Separate diagnostic builds from throughput builds. Record invalid-state,
  floor/reset, and fallback-correction counts in diagnostic builds when needed.
  Do not time added counters or synchronous instrumentation as production code.
- The shortened blast benchmark exits nonzero on its final kinetic-energy
  check. Accept that only after verifying the log shows the known early-stop
  failure, not a crash or numerical failure. Require a complete relevant
  regression test before retaining any proposed fix.

For each variant, save a manifest row: source base, patch, dependency hashes,
compiler flags, executable hash, log paths, throughput samples, numerical
checks, and interpretation. Compare **time differences**, not additive speedup
percentages.

## 3. Map and partition the commit

Inspect the complete parent-to-commit diff and map each changed call site to the
GPU kernel that executes it. Verify which branches are active in historical
HydroBlast3D, including `reconstruct_eint`, scalar counts, and the EOS state type.
Do not spend benchmark runs on changes compiled out of this problem.

Use these initial groups:

| Group | Changes to isolate |
|---|---|
| A: primitive conversion and validity | EOS pressure in `ConservedToPrimitive` and `CheckStatesValid`; helper calls replacing already-available thermal energy |
| B: timestep/wave-speed calculations | `maxSignalSpeedLocal`, `ComputeMaxSignalSpeed`, and the new conserved-state `ComputeSoundSpeed` helper |
| C: shock-flattening coefficients | `gamma * P` replaced by `rho * pow(EOS::ComputeSoundSpeed(...), 2)`; neighbor pressure conversions only if active |
| D: interface flux states | EOS pressure, sound speed, and energy calls for left/right states in `ComputeFluxes` |
| E: enforcement of floors | EOS pressure and pressure-to-energy calls in `EnforceLimits` |
| F: supporting configuration | Runtime gamma initialization, finite molecular-weight defaults, EOS wrappers/state types, and relevant problem setup |

Changes to other problem executables cannot explain HydroBlast3D timing unless
they affect shared compiled code. Document and exclude such changes. Check
explicitly for unrelated launches, allocations, synchronization, or numerical
algorithm changes rather than assuming the commit title describes every hunk.

## 4. Locate the expensive groups with reciprocal ablations

1. On the slow revision, restore the parent's behavior for groups A–E one at a
   time. Leave the supporting EOS machinery available so unrelated calls remain
   unchanged. Measure each against the untouched slow control.
2. Rank groups by recovered time per update and matching kernel-time changes.
   Combine implicated groups to test whether they recover the complete gap.
3. Check interactions: a group can matter only in combination with another.
   If individual removals do not explain the loss, bisect groups in combinations
   and test pairwise interactions among candidates. Do not assume additivity.
4. Confirm causality in the opposite direction: add the implicated changes to
   the fast parent and check that they reproduce the corresponding loss.
5. When forward ports need finite molecular weight or runtime gamma setup,
   introduce those prerequisites in a separately measured “parent + support”
   control. Use a consistent valid EOS state in every relevant variant; do not
   use NaN molecular weight as an artificial fast path.
6. Split any implicated group into individual call sites and expressions until
   the remaining responsible patch is small and reviewable.

Start with whole-application single-GPU measurements. Confirm leading candidates
and the combined reconstruction on a full node in the same allocation. If the
ranking changes between configurations, preserve and explain that distinction.

## 5. Isolate the mechanism inside the implicated historical code

Perform only controls justified by the group-level results:

### EOS arithmetic versus its interface

- Keep the historical Quokka call sites and wrapper, but simplify the historical
  gamma-law arithmetic. Port only the relevant formulas; do not replace the
  dependency wholesale with today's version.
- Separately test direct ideal-gas arithmetic through the same Quokka interface.
  Match runtime gamma and validation/reset behavior for the primary comparison.
- Use a wrapper-bypass build only as a diagnostic bound. It changes behavior for
  invalid inputs and cannot serve as a production fix or an equivalent result
  without proof that those paths are irrelevant to the measured workload.
- Compare runtime gamma against an equal compile-time gamma as a separate
  diagnostic. Attribute its benefit separately from algebraic simplification.

### Quokka call patterns and repeated work

- Pass already-computed rho, pressure, and thermal energy into the EOS instead
  of reloading conserved fields and recomputing kinetic energy.
- Compare separate sound-speed/energy calls against one EOS evaluation whose
  outputs are reused, preserving inputs and validation semantics.
- Inspect whether repeated mass-scalar construction survives compilation for
  this problem. Do not assume source-level duplication executes on the GPU.

### Shock flattening

If group C matters, compare the parent's `gamma * P`, the committed
`rho * cs^2`, and a direct calculation of `rho * cs^2` that avoids a square root
followed by squaring. Control runtime gamma and EOS floor semantics separately.
This distinguishes the caller's requested quantity from the implementation of
sound speed itself.

### Numerical and compiler effects

- Check whether new validation changes the detection of negative pressure,
  floor handling, or fallback frequency; measure this rather than assuming
  algebraic equivalence implies identical decisions.
- For affected historical kernels, compare GPU ISA and resource metadata:
  executed division/sqrt paths, scalar/global loads, duplicated calculations,
  branches, registers, spills, occupancy limits, and function calls.
- Collect per-kernel GPU profiles for fast, slow, and decisive ablation binaries
  with identical profiler settings. Separate setup from steady-state timesteps.
  Use unprofiled runs for throughput claims.
- Use hardware counters only when needed to distinguish a specific competing
  explanation, such as instruction throughput versus memory traffic. Register
  counts alone do not prove occupancy is limiting; managed storage alone does
  not prove page migration.
- Use a small standalone kernel only to reproduce a mechanism already observed
  in the historical application. Its speedup is supporting evidence, not a
  substitute for application-level recovery.

## 6. Closure criteria and deliverables

Call the regression explained only when:

1. The historical endpoint gap is reproducible under controlled conditions.
2. Removing a minimal set of changes from the slow endpoint recovers the loss,
   and adding them to the fast endpoint reproduces it, within measurement noise.
3. Kernel-level measurements and generated code support a concrete mechanism.
4. Numerical behavior and amount of executed work are accounted for.
5. The attribution holds on the full-node configuration relevant to the original
   report, or the single-GPU/full-node difference is explicitly explained.

Deliver a Markdown report with the responsible hunks, reciprocal measurements,
interaction results, GPU evidence, numerical checks, limitations, and exact
reproduction commands. Save every experimental patch and log. If no minimal
set explains the full gap, report the unexplained remainder instead of claiming
completion. Propose a production fix only after the attribution is established;
validate it separately before considering a port to current Quokka.

**This document authorizes no execution. Creating it does not resume the paused
performance investigation.**
