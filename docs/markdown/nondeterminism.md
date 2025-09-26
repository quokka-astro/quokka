# Sources of Nondeterminism

Quokka strives for reproducible physics, but some parts of the solver rely on parallel reductions that are not bitwise deterministic. The final double precision values can vary by a few units in the last place from run to run, especially when the MPI domain decomposition or thread scheduling changes. This page summarizes the current sources of nondeterministic behavior and offers guidance for interpreting results.

## Particle deposition

Particle-based modules deposit conserved quantities onto the mesh using parallel reductions. Each process or GPU thread accumulates contributions into shared mesh cells, and the use of floating-point arithmetic means the ordering of additions matters. The deposition algorithm incorporates stochastic rounding to keep conserved quantities unbiased, which makes the results reproducible in practice while still admitting a small probability that the rounding path differs between runs. When this rare event occurs, the discrepancies remain at roundoff level and do not grow without bound, but they prevent bitwise identical outputs across runs. The chance of seeing a deviation scales with the number of independent rounding events, roughly

\[
P_{\mathrm{nonrep}} \lesssim N_{\mathrm{round}} \times 2^{-R},
\]

where $N_{\mathrm{round}}$ counts mesh sites that trigger the roundoff kernel and $R$ is the `particles.reproducibility_roundoff_redundancy` setting (default $R=20$).

**Mitigation tips:** keep the domain decomposition fixed when comparing runs, and favor scalar diagnostics (e.g., total mass) that are less sensitive to rounding noise. For regression testing, compare against tolerances instead of strict bitwise equality, and note any stochastic rounding effects in experiment logs.

## Feedback routines

Stellar feedback and similar physics modules update grid cells using atomic operations so that multiple particles or mesh patches can safely modify the same state. Atomics guarantee correctness, and the feedback rounding procedure is designed to be reproducible, but it shares the same stochastic rounding mechanism as the deposition step. Consequently, results are stable across repeated runs with high probability while still carrying a small chance that rounding tie-breaks differ and introduce bit-level variations. The same estimate applies: with $N_{\mathrm{round}}$ feedback updates, deviations occur with probability no larger than $N_{\mathrm{round}} 2^{-R}$.

Whenever strict reproducibility is required, re-run with feedback disabled and document the change alongside the results, including any stochastic rounding considerations.

## AMR refluxing

When adaptive mesh refinement (AMR) is active, Quokka performs a reflux step to synchronize fluxes between coarse and fine levels. The reflux accumulator uses atomic additions to combine fluxes from neighboring patches. The summation order again depends on the hardware and parallel execution path, resulting in nondeterministic corrections applied to the coarse grid. These differences are bounded by the local truncation error of the solver but still break bitwise reproducibility.

Users should expect small, mesh-level differences between repeated AMR runs. Downstream diagnostics—such as global conservation errors or integrated luminosities—remain reliable, but they should be compared using relative/absolute tolerances rather than exact equality.

## Summary and recommendations

- All currently known nondeterministic paths stem from floating-point reductions guarded by atomic operations.
- The effect is typically limited to ~1e-13 relative differences in conserved quantities, though localized structures may drift slightly in position.
- For troubleshooting or regression checks, prefer norm-based comparisons (`L1`, `L2`, or `L_infty`) or diagnostic time series over direct file diffs.
- Document solver settings (AMR hierarchy, number of MPI ranks, GPU/CPU configuration) alongside published results so others can reproduce the experiment within expected tolerances.
