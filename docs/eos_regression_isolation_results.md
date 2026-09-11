# Isolation of the July 2023 EOS performance regression

## Conclusion

The regression in `efc47cf90657e1eee74bb4da01e5b3c0f0597c3a` is caused by replacing
cheap gamma-law expressions with general EOS queries in the hydro kernels.
It reproduces with dependency updates held fixed. Setup-only controls rule out
molecular-weight and gamma initialization as the main cause, and no unrelated
hydro algorithm bundled into the commit explains the measured loss.

An eight-expression reversal, confined to four functions in `src/hydro_system.hpp`,
recovers approximately 98% of the extra timestep time while leaving the rest of
the commit intact. In one allocation with three randomized measured repetitions:

| Variant | Median Mzone-updates/s | Range |
|---|---:|---:|
| Parent with the commit's EOS support/setup | 1130.799 | 1129.656–1131.311 |
| Regressing commit | 1020.612 | 1014.221–1026.253 |
| Eight-expression reversal | 1128.912 | 1128.790–1129.406 |

These are **full-application, full-node measurements**, not synthetic EOS timings.
The remaining roughly 0.17% throughput difference is much smaller than the
original 9.7% loss. A final matched allocation checked the two remaining EOS
expressions in `EnforceLimits`:

| Variant | Median Mupdates/s | Range |
|---|---:|---:|
| Parent support control | 1126.509 | 1124.300–1126.693 |
| Eight-expression reversal | 1122.414 | 1121.542–1123.853 |
| Ten-expression reversal, including `EnforceLimits` | 1124.803 | 1123.597–1125.099 |

The ten-expression reversal restores throughput within the control's observed
run-to-run spread. Its roughly 0.15% median difference is not resolved as a separate
regression. The [ten-expression patch](../build/eos-commit-isolation/minimal-all/complete.patch)
adds direct pressure and pressure-to-energy formulas in `EnforceLimits` to the
four-function patch above. Both reversal variants produce final fields exactly
identical to the parent's in the 256^3, 100-step comparison.

The reciprocal evidence is also positive. Forward-porting the relevant hydro
changes onto the parent plus EOS support produced 1019.887 Mupdates/s, versus
1130.140 for its control. Reapplying the eight expressions to the reversed source
reconstructs the tested slow source exactly; its other source files and dependencies
are unchanged. See the [eight-expression patch](../build/eos-commit-isolation/minimal-revert/complete.patch),
[forward patch](../build/eos-commit-isolation/minimal-revert/forward-eight.patch), and
[reciprocity manifest](../build/eos-commit-isolation/minimal-revert/reciprocity.json).

## The responsible expressions

| Function | Commit's added work | Fast expression restored by the control |
|---|---|---|
| `HydroSystem::ComputePressure` | General EOS query after obtaining conserved thermal energy | `(gamma - 1) * thermal_energy` |
| `HydroSystem::ComputeSoundSpeed` | General pressure query followed by a general sound-speed query | Direct pressure followed by `sqrt(gamma * P / rho)` |
| `ComputeFlatteningCoefficients` | `rho * pow(EOS::ComputeSoundSpeed(rho, P), 2)` | `gamma * P` |
| `ComputeFluxes` | Separate sound-speed and pressure-to-energy EOS queries for both interface states | `sqrt(gamma * P / rho)` and `P / (gamma - 1)` on each side |

The pressure helper also changes `AddInternalEnergyPdV`, even though that caller's
source line did not change. Reverting primitive-conversion code separately is not
necessary once the shared helper is restored: `revert-BCDG` measured 1129.224
Mupdates/s in the same allocation. This is why inspecting only visibly changed
kernel bodies would miss part of the regression.

The eight-expression control keeps the new helper structure, empty mass-scalar
construction, finite molecular weight, runtime gamma initialization, and inactive
reconstruction branches. It therefore rules out those structural changes as the
main cause. It is a diagnostic for this constant-gamma problem, not a general-EOS
replacement suitable for all Quokka configurations.

## What makes these EOS calls expensive

### Dynamic EOS bounds are a substantial cost

The wrapper loads runtime density, temperature, pressure, and energy bounds,
clamps/checks inputs, and retains the fallback rho-temperature calculation.
Replacing those bounds with the **same verified values** while preserving every
check and reset raised throughput from 1020.612 to 1081.150 Mupdates/s in the
same allocation. Gamma remained a runtime parameter. Host assertions verified
that the specialized bounds matched the initialized values.

This is evidence for the cost of dynamic parameter access and the compiler
simplifications it prevents. It does not separately establish how much is load
latency versus constant propagation. In particular, GPU-managed declarations
alone do not prove page migration or a particular NUMA-placement mechanism.

A further control combines fixed bounds, preserved EOS inputs, direct `dpde`,
and multiplication instead of `pow`, still retaining runtime gamma and validation.
It reached 1090.270 Mupdates/s. In its 20-step GPU profile, the main affected kernel
groups returned close to the parent's timings. Its 100-step throughput did not
fully recover, so this result is not presented as a complete optimized replacement.

The [bounds-only patch](../build/eos-commit-isolation/constant-bounds/complete.patch)
and [combined control](../build/eos-commit-isolation/specialized-runtime-gamma/complete.patch)
are specific to the verified benchmark configuration; runtime changes to those
bounds require the general implementation.

### Redundant thermodynamic arithmetic contributes, but is not the whole explanation

The original gamma-law implementation recovers supplied pressure through
pressure → temperature → pressure, and energy through energy → temperature →
pressure → energy. The callers also convert energy per volume to specific energy
and back. Without floating-point reassociation, mathematically cancelling these
operations is not generally legal.

Preserving supplied pressure/energy in the historical EOS improved throughput to
1029.702 Mupdates/s in its measurement cohort. Replacing runtime gamma with 1.4,
as a separate diagnostic, measured 1026.669. Neither alone recovers the regression.
These controls and their ranges are in the full results table; do not add their
percentage speedups or compare medians from different allocations as if paired.

The earlier synthetic derivative benchmark was particularly misleading as an
explanation of this commit: the 2023 hydro path has no `ComputeOtherDerivatives`
call. Its large derivative-only speedup concerned current Quokka's later HLLC
implementation, not the historical changed path measured here.

### Shock flattening introduces an expensive general power operation

The historical binary does not reduce `pow(cs, 2)` to a cheap multiply. The HIP
math header routes double `pow` to `__ocml_pow_f64`, and the affected kernel
contains a large math sequence. Changing only this expression to `cs * cs`,
while retaining the same EOS query, reduces the x-direction kernel's reported
VGPR allocation from 120 to 48; the parent uses 32. There are no reported scratch
allocations for these kernels.

The separate forward control is useful: in its matched cohort, the parent support
control measured 1125.270 Mupdates/s, introducing the flattening change measured
1079.894, and using multiplication with the new EOS call measured 1104.359.
Thus both the EOS query and the squaring expression contribute. Register counts
support the code-generation explanation, but do not alone prove a particular
occupancy or bandwidth limit.

The [squaring-only patch](../build/eos-commit-isolation/square-only/complete.patch)
is a numerically validated local optimization. It does not fix the entire EOS
regression.

### Both interface-state queries matter

Forward controls separately introduced the sound-speed and pressure-to-energy
queries in `ComputeFluxes`. In the same three-round cohort:

| Variant | Median Mupdates/s | Range |
|---|---:|---:|
| Parent support | 1125.270 | 1123.761–1126.834 |
| Sound-speed queries only | 1082.659 | 1080.429–1085.380 |
| Energy queries only | 1075.451 | 1072.594–1089.221 |
| Both query types | 1041.738 | 1022.415–1082.302 |

The combined effect is variable and cannot be apportioned by simply adding
individual time differences. Group reversions and forward additions nevertheless
agree that these interface-state queries are a major cause.

### Reusing caller work did not establish another major cause

A matched three-round allocation measured 1013.991 Mupdates/s for the slow
control (range 982.545–1027.250), 996.944 for reusing already-computed conserved
quantities (988.607–1032.761), and 1022.122 for obtaining flux-state energy and
sound speed from one EOS evaluation (1012.048–1025.581). These overlapping ranges
do not establish a throughput improvement. The loss is tied to introducing the
EOS computations, not simply the existence of a helper or two source-level calls.
No precise percentage of the loss is assigned to duplicated caller calculations.

## Kernel-level evidence

Twenty timesteps, summed GPU kernel durations across eight ranks:

| Kernel group | Parent (s) | Slow commit (s) | Eight-expression reversal (s) |
|---|---:|---:|---:|
| `ComputeFluxes` | 3.2477 | 4.4100 | 3.1884 |
| `ComputeFlatteningCoefficients` | 0.3970 | 1.3434 | 0.3980 |
| `ConservedToPrimitive` | 0.4963 | 0.8003 | 0.4902 |
| `AddInternalEnergyPdV` | 0.4557 | 0.8209 | 0.4551 |
| `EnforceLimits` | 0.5790 | 0.5796 | 0.5641 |

Parent/slow profiles share one allocation; the reversal profile is from the
closure allocation. They support location and recovery of the cost, rather than
an exact additive decomposition of elapsed timestep time. Unprofiled repeated
runs provide the throughput claims. Profiler warnings about unsupported SPM were
nonfatal: all expected rank kernel CSVs were produced.

EOS code is inlined in the inspected kernels; species and entropy work are absent
for this problem's `chem_eos_t`. The static assembly retains full FP64 division
sequences, runtime parameter loads, and reset branches. Thus “function-call
overhead” or “computing the complete EOS including entropy” is not the explanation.

## Numerical controls and exclusions

- All safe full-node variants retained the same printed timestep history and
  kinetic energy (`0.01778536047` after 100 steps on 512^3), with energy conservation.
- Complete 256^3 final plotfiles were compared using the historical AMReX
  `fcompare`, covering all six evolved fields. Parent/slow, reciprocal controls, both compact reversals,
  squaring, arithmetic, fixed bounds, and combined specialization pass relative
  tolerance 1e-11 and absolute tolerance 1e-12. The original parent/slow maximum
  field-norm relative differences are around 3e-15.
- The complete historical 128^3 Sedov test passed for the squaring-only change,
  including the kinetic-energy check (relative error -0.006891676163), exit code 0.
- The shortened timing runs intentionally fail the final kinetic-energy reference
  check because they stop early. Their exit code 1 is accepted only with completed
  evolution and passing energy conservation; it is not treated as a passing
  complete regression test.
- Bypassing the wrapper produced roughly 34% energy loss on the local benchmark.
  Its apparent throughput is excluded from equivalent-performance claims.
  A 32^3, 10-step diagnostic recorded 32,765 rho-energy resets, confirming that
  this path actually executes; those counts are not extrapolated to the large run.
- The profile has the expected two-stage, three-direction flux-kernel counts,
  with no extra fallback flux evaluations in the endpoint comparison. The measured
  regression is not additional FOFC work or a different number of zone updates.

## Experimental scope and reproducibility

Only the July source change is under test. Both endpoints use AMReX
`de7c6189623bc3ba178a380e703e3173f33f78ef`, Microphysics
`ced7cc41dbafae11ed587c1d446d80e442b9b967`, and fmt
`e8259c5298513e8cdbff05ce01c46c684fe758d8`. The same saved compatibility patches,
ROCm 7.14/LLVM 23 compiler, gfx90a target, Release flags, and existing dependency
libraries are used. The native submodule bump is therefore excluded. No result
here tests the later September FOFC/RK2 rewrite.

The benchmark is HydroBlast3D, gamma 1.4, no chemistry or mass scalars,
`reconstruct_eint=false`, octant geometry, 100 timesteps. The full node uses eight
ranks, seven CPUs/rank, one GPU/rank and closest binding. Runs include warmups and
three randomized repetitions per variant. Compare medians and ranges within each
allocation. The slow/EOS variants show substantial process-to-process variation;
local MI210 measurements are especially variable and are not scaled by eight.
The cause of that variability has not been established. It does not prevent the
stable full-node reverse/forward result, but limits precise attribution of small
individual effects.

Artifacts under [`build/eos-commit-isolation`](../build/eos-commit-isolation):

- [Complete measurement table](../build/eos-commit-isolation/RESULTS_TABLE.md) and
  [machine-readable results](../build/eos-commit-isolation/node-results.json),
  including warmup labels, ranges, energy checks, log hashes and log paths.
- Per-variant `manifest.json`, `compile.json`, `link.json`, `complete.patch`, source
  snapshots, object logs and executable. `environment.json` and the common
  compatibility patches record the build environment.
- Raw per-rank profiler CSVs, GPU code/assembly, final plotfiles, comparison logs,
  reset diagnostics and the full regression log.
- `build_variant.py`, `prepare_*.py`, `run_local.py`, `run_fields.py`,
  `compare_fields.py`, `profile_summary.py`, `save_patches.py`,
  `summarize_results.py`, and the submitted Slurm scripts.

Representative commands, from the checkout with the recorded ROCm environment:

```bash
python3 build/eos-commit-isolation/build_variant.py parent aa288b57
python3 build/eos-commit-isolation/build_variant.py slow efc47cf9
python3 build/eos-commit-isolation/run_local.py parent slow parent slow slow parent parent slow
sbatch build/eos-commit-isolation/closure-node.submit
python3 build/eos-commit-isolation/summarize_results.py
```

Preparation scripts construct the named variants and their overrides; the builder
preserves an already-prepared source tree. Use the saved patches/manifests to
verify which variant is being rebuilt, rather than assuming a directory name
alone identifies its contents.

## Completion and next implementation boundary

The source-level attribution is complete: fresh frozen-dependency endpoints,
individual reversions, forward additions, combined interactions, compact reversals,
mechanism controls, per-kernel profiles, full-field comparisons, and the complete
Sedov test are recorded. The [completion audit](../build/eos-commit-isolation/COMPLETION_AUDIT.md)
maps these results to the saved plan. The experiment includes 149 full-node timing
runs, including warmups and invalid diagnostic controls; only explicitly identified
valid repeated cohorts support the numerical comparisons above.

No production source was changed by this isolation run. The pre-existing uncommitted
Microphysics arithmetic patch is preserved. A production fix should specialize the
constant-gamma path, preserve necessary floor/reset behavior, avoid general `pow`
for squaring, and expose fixed EOS parameters to the compiler where valid. Porting
or benchmarking that design on current Quokka is separate work: the later FOFC/RK2
implementation was deliberately excluded here, and these historical results do not
promise a recovery of today's entire slowdown.
