# Sources of Nondeterminism

Quokka strives for reproducible physics, but some parts of the solver rely on parallel reductions that are not bitwise deterministic. The final double precision values can vary by a few units in the last place from run to run, especially when the MPI domain decomposition or thread scheduling changes. This page summarizes the current sources of nondeterministic behavior and offers guidance for interpreting results.

## Particle deposition

Particle-based modules deposit conserved quantities onto the mesh using parallel reductions. Each process or GPU thread accumulates contributions into shared mesh cells, and the use of floating-point arithmetic means the ordering of additions matters. Different task scheduling, MPI ranks, or GPU execution paths can therefore lead to small differences in the deposited mass, momentum, or energy. These discrepancies are typically roundoff level and do not grow without bound, but they prevent bitwise identical outputs across runs.

**Mitigation tips:** keep the domain decomposition fixed when comparing runs, and favor scalar diagnostics (e.g., total mass) that are less sensitive to rounding noise. For regression testing, compare against tolerances instead of strict bitwise equality.

## Feedback routines

Stellar feedback and similar physics modules update grid cells using atomic operations so that multiple particles or mesh patches can safely modify the same state. Atomics guarantee correctness, but they do not enforce a deterministic ordering of updates. On CPUs this depends on the thread scheduler, and on GPUs it varies with the kernel launch configuration. As a result, runs with feedback enabled may diverge down to the last few bits even if every other setting is identical.

Whenever strict reproducibility is required, re-run with feedback disabled and document the change alongside the results.

## AMR refluxing

When adaptive mesh refinement (AMR) is active, Quokka performs a reflux step to synchronize fluxes between coarse and fine levels. The reflux accumulator uses atomic additions to combine fluxes from neighboring patches. The summation order again depends on the hardware and parallel execution path, resulting in nondeterministic corrections applied to the coarse grid. These differences are bounded by the local truncation error of the solver but still break bitwise reproducibility.

Users should expect small, mesh-level differences between repeated AMR runs. Downstream diagnostics—such as global conservation errors or integrated luminosities—remain reliable, but they should be compared using relative/absolute tolerances rather than exact equality.

## Summary and recommendations

- All currently known nondeterministic paths stem from floating-point reductions guarded by atomic operations.
- The effect is typically limited to ~1e-13 relative differences in conserved quantities, though localized structures may drift slightly in position.
- For troubleshooting or regression checks, prefer norm-based comparisons (`L1`, `L2`, or `L_infty`) or diagnostic time series over direct file diffs.
- Document solver settings (AMR hierarchy, number of MPI ranks, GPU/CPU configuration) alongside published results so others can reproduce the experiment within expected tolerances.
