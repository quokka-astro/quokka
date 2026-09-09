# Nonlinear reconstruction failure in 2D shear

## Positive-candidate fallback

The current implementation rejects nonpositive final xPPM edges for hydro density,
the pressure/specific-internal-energy component, the auxiliary thermodynamic
component, and all passive scalars. Each rejected edge is replaced with its corresponding `MonotonizeEdges`
value. Positive edges and all signed components retain their original results.
An explicit component policy is passed through the reconstruction overloads; other
callers default to unrestricted signed reconstruction. No new physical or relative
floor is introduced. This protects reconstructed face values for admissible positive
input data, not the entire subsequent Euler update.

`HydroShearRepro_positive_reconstruction` exercises the captured bad stencil in
native reconstruction and checks that the invalid edge uses the positive fallback,
its positive partner is unchanged, signed components are unchanged, and an
admissible shifted stencil remains bitwise unchanged. A two-scalar policy fixture
also checks both passive scalar components and excludes the component immediately
after the scalar range. The comparison script now
requires all six runs to complete by default. Use `--expect-original-failure` only
with an original executable to check the historical outcomes below.

The native 2D CPU comparison passes all six cases to t=3 (PPM and xPPM at
CFL=0.3/AV=0, CFL=0.15/AV=0, and CFL=0.3/AV=0.1), with dual energy enabled
throughout. `HydroShearRepro_xppm_av0` registers the formerly crashing case as a
successful CTest regression. Logs, commands, and executable hashes are saved in
`build/2d/shear-positive-fallback/`. GPU and multi-rank execution were not tested.

### Rejected rescaling experiment

An earlier implementation contracted both edges toward the donor average with a
bound of `1e-12` times that average. Although its captured-stencil positivity test
passed, it did not cure the full 32 × 32 case: xPPM at CFL 0.3 exhausted retries in
the first coarse step with final rejected signal speed 2626.144. At CFL 0.15 it
failed after the last completed coarse time 0.00600089; that control had previously
completed. Both PPM controls and both AV=0.1 controls completed to t=3. The mechanism
of that new failure was not traced. This rescaling has been replaced by candidate
rejection, rather than retained as a solver option. Its logs remain in
`build/2d/shear-positivity-final/`.

## Historical reproducer before positive-candidate fallback

`HydroShearRepro` supplies a small, source-free compressible Euler problem that distinguishes classic PPM from xPPM in the current solver. The failure input is `inputs/HydroShearReproFailure.toml`. Every comparison enables dual energy and uses the existing SSP-RK2 integrator, nonlinear reconstruction, shock flattening, HLLC flux, first-order flux correction, and timestep retries. No solver algorithm was changed to obtain the failure.

The problem is motivated by high-Mach vortex-sheet interaction tests, such as §3.2.2 of [Pan, Li & Xu (2017), *A Few Benchmark Test Cases for Higher-Order Euler Solvers*](https://www.math.hkust.edu.hk/~makxu/PAPER/Benchmark-CFD-inviscid.pdf). The particular oblique, perturbed shear below is a separate constructed test, not a reproduction of that paper's initial conditions.

## Initial conditions

Use a periodic unit square, a uniform **32 × 32** mesh, and an ideal gas with gamma 1.4. Define

\[
\phi=2\pi(x+3y),\qquad \psi=2\pi(-3x+y),\qquad
f=\frac{\tanh(30\sin\phi)}{\tanh 30}.
\]

The initial state is

\[
\rho=1+0.9999f,\qquad P=2\times10^{-7},\qquad
\mathbf v=f\frac{(-3,1)}{\sqrt{10}}
+0.3\sin\psi\frac{(1,3)}{\sqrt{10}}.
\]

The pressure is the highest value in the tested sweep that reproduces the xPPM failure; it is not an optimized stability boundary. The sampled initial Mach numbers range from approximately 18.9 to 2790.

Density varies between 0.0001 and 1.9999: a contrast of 19999:1. The thin layers are intentionally underresolved. Both velocity terms have zero analytic divergence; the flow has vorticity and is not aligned with the mesh. The transverse perturbation causes nonlinear evolution, so there is no prescribed final reference solution. Initial conserved states are sampled at cell centers, as in other Quokka problem drivers.

There is no gravity, cooling, heating, physical viscosity, radiation, MHD, or particle physics. The density floor is `1e-12`; the temperature floor is `1e-20` in this problem's units. The temperature parameter is not a pressure floor. Artificial viscosity is varied explicitly below. All tests use `hydro.use_dual_energy=1`.

The more general `inputs/HydroShearRepro.toml` supplies a mild shear. Runtime `shear.*` parameters control density amplitude, pressure, shear amplitude, integer mode components, sharpness, transverse perturbation, and bulk velocity. With zero perturbation, the profile is an exactly advected shear in the continuum. The driver requires a periodic unit square and no AMR, and is compiled only in 2D.

## Reproduce and compare

From the repository root, with a configured 2D build:

```bash
./scripts/bash/quokka build -d 2d HydroShearRepro
python3 src/problems/HydroShearRepro/run_reconstruction_comparison.py
```

The script uses isolated run directories under `build/2d/shear-reconstruction-comparison/`, records the complete argument vector and executable/input SHA256 hashes, and writes `results.json`. It checks the specific `Hydro update exceeded max_retries` failure, rather than treating an arbitrary nonzero exit as a reproduction. Successful runs must reach the requested final time. It accepts `--exe`, `--input`, `--output`, `--stop-time`, and `--timeout`.

The expected-failure run deliberately aborts through Quokka's existing retry limit. To run the matched pair manually, use separate output directories and absolute paths to the executable and input:

```bash
HydroShearRepro HydroShearReproFailure.toml hydro.reconstruction_order=3
HydroShearRepro HydroShearReproFailure.toml hydro.reconstruction_order=5
```

## Measured behavior

Native AppleClang Release CPU build, double precision, one MPI rank, 2D, base commit `c8e0bcce7178c63768452eaf893a495ade8b1fb7` plus the problem driver:

| Reconstruction | Artificial viscosity coefficient | CFL | Result at requested t = 3 |
|---|---:|---:|---|
| PPM (order 3) | 0 | 0.3 | Completes |
| xPPM (order 5) | 0 | 0.3 | Exhausts retries in first coarse timestep |
| PPM | 0 | 0.15 | Completes |
| xPPM | 0 | 0.15 | Completes |
| PPM | 0.1 | 0.3 | Completes |
| xPPM | 0.1 | 0.3 | Completes |

The failing xPPM run begins with a coarse timestep of approximately 0.008546 and maximum signal speed 1.097. Rejected attempts report maximum signal speeds 10.639, 3.357, 9.534, 9.940, 24.753, 41.047, and 100.447. The final attempt violates the post-update CFL check even at dt/64. First-order flux correction also activates near the end. The run then prints:

```text
QUOKKA FATAL ERROR
Hydro update exceeded max_retries on level 0. Cannot continue, crashing...
```

Some smaller substeps were accepted before failure; “first coarse timestep” does not mean the state never evolved. Retrying preserves accepted substeps, so starting a separate run at smaller CFL follows a different trajectory. PPM also retries (57 retry messages over t = 3 in the baseline run), but completes.

Native tracing and snapshot replays identify a specific cause of the terminal 32 × 32 failure: xPPM's final median restores a negative reconstructed density after monotonization repaired it. HLLC receives that invalid face state and produces a large energy flux, raising the adjacent cell's sound speed above the post-update CFL limit. See the [cell-by-cell diagnosis](shear_reconstruction_diagnosis.md) for the stencil, limiter branches, flux accounting, and causal interventions. This does not establish entropy instability or smooth-stencil RK2 amplification as the cause. Negative pressure inferred from total energy alone is not used as a failure criterion; dual energy remains enabled throughout.

The difference is parameter-dependent: at 64 × 64 both methods fail with the current pressure `2e-7`; at 32 × 32 with initial pressure raised to `1e-6`, both complete. Completion is a robustness diagnostic, not evidence of converged accuracy. GPU and multi-rank behavior have not been validated.

### Resolution 64 × 64

Changing only the mesh to 64 × 64 in the current reproducer (pressure `2e-7`,
density contrast 19999:1, CFL 0.3, AV off, dual energy enabled) makes **both
PPM and xPPM exhaust retries during the first coarse timestep**. Both start
with dt = 0.00427323 and still violate the post-update CFL check at dt/64 =
0.0000667692. The final rejected attempts have maximum signal speeds of
83.13 for PPM and 80.98 for xPPM. Some substeps are accepted, but neither
finishes the first coarse step. The PPM/xPPM separation observed at 32 × 32
therefore does not persist at this resolution. Commands, logs, and results
are in `build/2d/shear-resolution64/`.

### Density contrast sweep

Using the earlier initial pressure `1e-8` and holding all other settings fixed
(32 × 32, CFL 0.3, AV off, RK2, dual energy enabled, final time 3), vary only the density amplitude
as `shear.density_amplitude = (R-1)/(R+1)` for a maximum-to-minimum density
ratio R. Both PPM and xPPM completed for R = 1, 1.01, 1.1, 1.5, 2, 3, 10,
100, 1000, 2000, 5000, and 10000. At R = 19999, PPM completed and xPPM again
exhausted its retries. Thus the demonstrated failure did not reproduce at
small initial density contrast in this sweep; these samples do not establish
a monotonic failure threshold. Commands, logs, and results are saved under
`build/2d/shear-density-contrast/`, including `sweep.py` and `results.json`.

At the updated pressure `2e-7`, a second sweep decreased R from 19999 by
successive factors of 1.5, continuing through R = 1.188 and adding a uniform
density control (R = 1). All other reproducer settings remained fixed,
including dual energy. This comprised 26 contrasts and 52 runs. Only the
original R = 19999 reproduced xPPM retry exhaustion with PPM completing.
At the next lower sampled contrast, R = 13332.6667, both methods reached
t = 3; both also completed at every remaining lower sampled contrast.
Thus 19999 remains the lowest reproducing contrast on this factor-1.5 grid,
not a proven continuous stability threshold. The input amplitude remains
`0.9999`. Commands, logs, the sweep script, and `results.json` are in
`build/2d/shear-density-factor15/`.

### Initial Mach number sweep

Holding the density contrast at 19999:1 and the velocity field fixed, varying
only the initial pressure scales every initial Mach number as P^(-1/2).
These comparisons again use 32 × 32, CFL 0.3, AV off, RK2, dual energy enabled,
and final time 3. The ranges below are evaluated at the actual cell centers;
the continuous profile has additional unsampled stagnation points.

| Initial pressure | Initial Mach range | PPM | xPPM |
|---:|---:|---|---|
| 1e-8 | 84.5–12478 | Completes | Exhausts retries |
| 2e-8 | 59.8–8823 | Completes | Exhausts retries |
| 5e-8 | 37.8–5580 | Completes | Completes |
| 1e-7 | 26.7–3946 | Completes | Exhausts retries |
| 2e-7 | 18.9–2790 | Completes | Exhausts retries |
| 5e-7 | 12.0–1765 | Completes | Completes |
| 1e-6 | 8.45–1248 | Completes | Completes |
| 1e-5 | 2.67–395 | Completes | Completes |
| 1e-4 | 0.845–125 | Completes | Completes |
| 1e-3 | 0.267–39.5 | Completes | Completes |
| 1e-2 | 0.0845–12.5 | Completes | Completes |
| 0.1 | 0.0267–3.95 | Completes | Completes |
| 1 | 0.00845–1.25 | Completes | Completes |
| 10 | 0.00267–0.395 | Completes | Completes |

The lowest maximum initial Mach number among the failing sampled cases is
about 2790. Failure is not monotonic in initial pressure, so this is not a
stability threshold. The more moderate-Mach cases tested completed; some
still required retries. Commands, logs, and results are in
`build/2d/shear-mach-sweep/`, including `sweep.py` and `results.json`.

## Diagnostics and regression controls

The history records time, minimum density, minimum pressure from auxiliary internal energy, mass, total energy, kinetic energy, maximum velocity magnitude, minimum raw pressure inferred from total energy, and the solver's maximum signal speed. It is written initially and after completed coarse timesteps. The failed run therefore has only its initial history row; the retry log records its internal evolution. Total energy changes can include the existing dual-energy synchronization and floors.

`HydroShearRepro_order3` and `HydroShearRepro_order5` register short successful controls with AV = 0.1 in CTest:

```bash
./scripts/bash/quokka run -d 2d --filter '^HydroShearRepro_'
```

The expected crash is exercised by the comparison script, not registered as a successful simulation in CTest. Its current expected outcome must be reconsidered if the solver is subsequently made more robust.
