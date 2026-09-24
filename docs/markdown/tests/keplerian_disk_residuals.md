# Keplerian disk equilibrium residual measurements

> These results describe the historical smooth annulus, not the current
> [Krumholz thin-ring setup](keplerian_disk.md). The analysis scripts retain the
> old analytic profile and domain; their reference comparisons do not apply to
> the current driver.

The initial imbalance is dominated by the spatial discretization in these
one-step experiments. Reducing the timestep does not remove either the
axisymmetric inward acceleration around the ring peak or the fourfold error.
This favors improving the spatial equilibrium balance before changing the
source time integration. It does not establish the cause or magnitude of
long-term accretion.

## Experiment

Measured on 2026-09-09 using the current checkout, 2D CPU, one MPI rank, uniform
128² and 256² grids, xPPM (`hydro.reconstruction_order=5`), dual energy enabled,
and explicit artificial viscosity coefficients K=0 and K=0.1. The input's
analytic equilibrium is sampled at cell centers, as in the existing driver.
No evolution algorithm was changed.

There are four independent one-step runs per resolution and viscosity setting:

- 128²: dt = 0.002, 0.001, 0.0005, 0.00025.
- 256²: dt = 0.001, 0.0005, 0.00025, 0.000125.

The optional `disk.residual_prefix` diagnostic requires a 2D run, one MPI rank,
and `max_timesteps=1`. It writes two cell-data files:

1. `*-spatial.txt`: evaluate the actual hydro numerical face fluxes on the
   initial equilibrium, take their divergence, and add analytic gravity to
   momentum. This includes HLLC and explicit artificial viscosity. It is
   evaluated before the first gravity kick.
2. `*-step.txt`: `(U_after - U_initial)/dt`, measured after the complete timestep,
   including both gravity kicks and both hydro RK stages.

Both contain `x y rho0 density_rate mx_rate my_rate torque_defect`.
The last column is only measured for the spatial residual; the step file has
a zero placeholder. Only mass and x/y momentum residuals are measured here,
not energy residuals.

For R = (R_rho, R_mx, R_my), the plotted quantities are

\[
R_\rho,\qquad
 a_r = \frac{xR_{m_x}+yR_{m_y}}{r\rho_0},\qquad
 \frac{R_L}{\rho_0}=\frac{xR_{m_y}-yR_{m_x}}{\rho_0}.
\]

Positive radial acceleration is outward. The third quantity is the angular
momentum **density** rate divided by initial density. It is not the time
derivative of specific angular momentum: that also has a density-rate term.
For the finite timestep, these are normalized finite differences, approaching
the corresponding instantaneous rates as dt tends to zero.

Profiles use annuli of width 2 dx. Within each annulus, least squares fits
`a0 + c4 cos(4 phi) + s4 sin(4 phi) + c8 cos(8 phi) + s8 sin(8 phi)`.
The fourfold amplitude is `hypot(c4,s4)`. Joint fitting avoids leakage from a
constant into m=4 due to nonuniform Cartesian angular sampling; unresolved
higher harmonics and radial variation within each annulus can still affect
the fitted coefficients. The JSON profiles also include the direct area mean.
The comparison at the ring peak uses the same annulus, 0.9375 <= r < 1.0625,
at both resolutions.

## Results

Area-weighted RMS residuals over cell centers with 0.7 < r < 1.3, in code units,
for K=0.1:

| Grid | Spatial density rate | Spatial radial acceleration | Spatial R_L/rho0 |
| --- | ---: | ---: | ---: |
| 128² | 1.62769e-3 | 6.61628e-3 | 1.01352e-2 |
| 256² | 2.71499e-4 | 1.94621e-3 | 1.66847e-3 |

The radial RMS falls by 3.40 and the angular-momentum RMS by 6.07 when doubling
resolution. These are two-grid measurements, not asymptotic order claims.

The timestep dependence at 128² and K=0.1 is:

| dt | One-step radial RMS | RMS of radial step-minus-spatial | One-step R_L/rho0 RMS |
| --- | ---: | ---: | ---: |
| 0.002 | 6.59088e-3 | 6.15836e-5 | 1.00754e-2 |
| 0.001 | 6.60329e-3 | 3.04052e-5 | 1.01049e-2 |
| 0.0005 | 6.60971e-3 | 1.52021e-5 | 1.01199e-2 |
| 0.00025 | 6.61298e-3 | 7.68150e-6 | 1.01275e-2 |
| Spatial limit | 6.61628e-3 | 0 | 1.01352e-2 |

The full residual approaches a substantial nonzero spatial limit. The
step-minus-spatial difference includes both hydro time integration and gravity
splitting; it does not isolate splitting alone. Its approximately linear
decrease with dt is not evidence of first-order time integration: even exact
time evolution gives `(U(dt)-U(0))/dt = L(U0) + O(dt)` when `L(U0)` is nonzero.

The radial acceleration around the ring peak has both a negative mean and a
fourfold variation:

| Grid | Direct area mean | Fitted a0 | Fourfold amplitude |
| --- | ---: | ---: | ---: |
| 128² | -1.47950e-3 | -1.49403e-3 | 1.38141e-3 |
| 256² | -3.76378e-4 | -3.74708e-4 | 3.46268e-4 |

Both fitted components decrease by approximately four with refinement.
The scalar spatial residual fields respect 90-degree rotational symmetry to
an absolute error below 4.4e-14 on both grids. Thus the fourfold pattern is
compatible with the square mesh's symmetry; there is no observed quadrant
asymmetry in this experiment. Eliminating that pattern alone would leave the
axisymmetric inward acceleration.

Turning K from 0.1 to 0 changes the spatial angular-momentum RMS at 128² from
1.013518e-2 to 1.013502e-2 (less than 0.002%). The radial RMS is unchanged at the
six significant figures shown. This does not disable HLLC's own corrections
or Riemann dissipation.

## Angular momentum and mass budgets

Gravity has zero torque cell by cell. For cell-centered
`L = x m_y - y m_x`, the initial numerical fluxes obey

\[
 R_L = -D_x(x_f F^x_{m_y}-y F^x_{m_x})
       -D_y(x F^y_{m_y}-y_f F^y_{m_x}) + T_h,
\]

where

\[
 T_h = \tfrac12(F^x_{m_y,i+1/2}+F^x_{m_y,i-1/2})
      -\tfrac12(F^y_{m_x,j+1/2}+F^y_{m_x,j-1/2}).
\]

The diagnostic computes T_h directly from the solver's face fluxes. Integrating
R_L and T_h gives the net outward face-moment flux via this exact discrete
identity. This is a budget on the stair-step boundary of selected Cartesian
cells, not an interpolated circular surface. It includes all numerical flux
terms, unlike the existing cell-centered advective transport profile.

Initial spatial budgets for cells with r < 1, K=0.1:

| Grid | dM/dt | dL/dt | Integral T_h | Outward face-moment L flux |
| --- | ---: | ---: | ---: | ---: |
| 128² | +5.05757e-5 | +4.68524e-5 | -3.34585e-6 | -5.01982e-5 |
| 256² | -6.62954e-6 | -5.37899e-6 | +1.23567e-6 | +6.61466e-6 |

Positive dM/dt means accumulation in that region. The initial mass rate inside
r=1 changes sign with resolution. Inside the regularized core, r<0.25, the
spatial mass rate is negative on both grids (-3.45558e-4 and -1.32585e-4).
Consequently these measurements show an initial equilibrium imbalance, not
sustained inflow to the origin. Long-time accretion requires a separate evolving
flux budget. The tabulated face-moment budget is for the initial spatial
operator; time-averaged RK face fluxes are not exported for the full step.

## Reproduction and validation

From the repository root:

```sh
./scripts/bash/quokka build -d 2d KeplerianDisk
python3 src/problems/KeplerianDisk/measure_residuals.py \
  --executable build/2d/src/problems/KeplerianDisk/KeplerianDisk
ctest --test-dir build/2d -R '^KeplerianDisk(Residual)?$' --output-on-failure
```

The script requires NumPy and Matplotlib. Omit `--executable` to reanalyze
existing data. Outputs are under `build/2d/keplerian-residuals/`:
`summary.json`, `residual-profiles.png`, `residual-profiles.pdf`, and per-run
raw residuals, profiles, `command.json`, history, and `run.log`.

All 16 runs completed. The analysis checks finite data, one-step timing,
whole-domain and inner-region mass/angular-momentum rates against independently
written histories, and recovery of manufactured m=0 and m=4 coefficients.
Both the original KeplerianDisk CTest and the new residual-mode CTest passed.
An additional 128², K=0.1, dt=0.002 control produced byte-identical history
and final advective transport files with the diagnostic enabled and disabled.
C++ formatting and `git diff --check` also passed.
Validation was on CPU with one MPI rank; this measurement mode deliberately
requires one rank and 2D.

## Exact-state controls: reconstruction versus integration

A follow-up experiment evaluates analytic equilibrium states on faces, without
changing the evolved state or evolution algorithm. It compares:

- **Exact midpoint:** identical analytic left/right states at each face center,
  with the existing flux-divergence formula and cell-center gravity.
- **Gauss integration:** identical analytic states at two or four quadrature
  nodes per face, averaging the flux returned by HLLC. Gravity is independently
  averaged over the cell using the corresponding tensor-product rule (2² or
  4² source evaluations).

Each HLLC evaluation is checked against the physical Euler mass and momentum
flux at that node. With identical states, the artificial-viscosity jump vanishes;
the pressure-jump term in the carbuncle correction also vanishes. These controls
therefore remove reconstruction errors and jump dissipation. The higher-order
control measures the finite-volume balance of analytic flux and source averages.
The exported `rho0` remains the original point-sampled density solely to keep
the normalization of the plotted residuals unchanged.

### Measured results

RMS over 0.7 < r < 1.3, using the same normalization and cell selection as above:

| Grid | Operator | Density rate | Radial acceleration | R_L/rho0 |
| --- | --- | ---: | ---: | ---: |
| 128² | Current xPPM, K=0.1 | 1.62769e-3 | 6.61628e-3 | 1.01352e-2 |
| 128² | Exact midpoint | 4.80940e-3 | 8.00604e-3 | 5.28235e-2 |
| 128² | Two-point Gauss | 1.03153e-5 | 1.12374e-5 | 6.87585e-5 |
| 128² | Four-point Gauss | 1.03032e-11 | 8.31362e-12 | 1.37185e-10 |
| 256² | Current xPPM, K=0.1 | 2.71499e-4 | 1.94621e-3 | 1.66847e-3 |
| 256² | Exact midpoint | 1.20844e-3 | 2.00684e-3 | 1.31890e-2 |
| 256² | Two-point Gauss | 6.49741e-7 | 7.17988e-7 | 4.31292e-6 |
| 256² | Four-point Gauss | 4.10020e-14 | 3.28612e-14 | 5.39627e-13 |

The exact-midpoint control is worse than current xPPM in all three norms on
both grids. Thus accurate point values at the face centers alone do not fix
this imbalance. This is not an upper bound on every wider reconstruction:
finite-volume reconstruction estimates derived from averages can have different
leading errors, and those errors can partially cancel integration errors.

Two-point quadrature reduces the radial residual by approximately 589 times
at 128² and 2711 times at 256² relative to xPPM. Its residuals decrease by about
16 with refinement, consistent with fourth-order integration in this smooth
annulus. Four-point quadrature reduces them much further. An independent
NumPy calculation with eight-point quadrature gives RMS residuals between
2.4e-15 and 1.1e-14, providing a roundoff-level reference.

In the common peak annulus, 0.9375 <= r < 1.0625:

| Grid | Operator | Radial a0 | Radial fourfold amplitude |
| --- | --- | ---: | ---: |
| 128² | Current xPPM | -1.49403e-3 | 1.38141e-3 |
| 128² | Exact midpoint | -2.27582e-3 | 1.90846e-3 |
| 128² | Two-point Gauss | +2.44459e-6 | 2.67332e-6 |
| 128² | Four-point Gauss | +9.87955e-13 | 6.44130e-13 |
| 256² | Current xPPM | -3.74708e-4 | 3.46268e-4 |
| 256² | Exact midpoint | -5.69860e-4 | 4.77184e-4 |
| 256² | Two-point Gauss | +1.52525e-7 | 1.65114e-7 |
| 256² | Four-point Gauss | +3.88513e-15 | 2.44350e-15 |

Both the axisymmetric inward acceleration and the fourfold component are
strongly reduced by matched quadrature. This experiment favors testing a
**generic multidimensional high-order flux/source discretization** before
building equilibrium-specific reconstruction. A wider normal PPM stencil alone
does not address the integration error demonstrated by the midpoint control.

These are analytic controls, not a production method. Reconstructing numerical
states at quadrature nodes, maintaining consistent cell averages and primitive
conversions, and limiting the resulting method remain unimplemented. The
experiment changes face and source quadrature together, so it does not assign
separate error fractions to each. It also does not measure long-time disk
survival, prove that a particular wider stencil cannot help, or remove the need
for matched time integration. Exact preservation would still require discrete
well-balancing. The core boundary is nonsmooth and is excluded from the quoted
smooth-annulus convergence results.

### Reproducing the controls

```sh
./scripts/bash/quokka build -d 2d KeplerianDisk
python3 src/problems/KeplerianDisk/compare_analytic_residuals.py \
  --executable build/2d/src/problems/KeplerianDisk/KeplerianDisk
```

The existing residual mode now additionally writes `residual-analytic-q1.txt`,
`residual-analytic-q2.txt`, and `residual-analytic-q4.txt`. Their torque-defect
column is a zero placeholder; no face-moment budget is claimed for these files.
The analysis omits that placeholder from reported summary metrics.

The comparison script creates `build/2d/keplerian-analytic-residuals/` with a
summary JSON, PNG/PDF comparison plots, and separate `n128/` and `n256/`
directories containing raw residuals, profiles, commands, and logs. Omit
`--executable` to reanalyze existing outputs.

Validation includes the HLLC/physical-flux checks at every quadrature node,
independent Python recomputation of all three control residuals (agreement
within 5e-13 in each mass/momentum rate component), and the eight-point
smooth-annulus reference. Both KeplerianDisk CTests passed. At both resolutions,
histories and final advective transport files are byte-identical to the prior
runs without these analytic controls. CPU, 2D, one MPI rank only; no GPU or
production higher-order evolution was tested.
