> Historical setup: this page describes the former smooth annulus. Its formulas,
> commands, and residual comparisons do not describe the current thin-ring driver.
> See [the current ring test](keplerian_disk.md).

# Keplerian disk and artificial viscosity

`KeplerianDisk` is a hydro-only annulus on a Cartesian mesh, designed to measure
numerical evolution of a steady rotating flow. It has no self-gravity, particles,
MHD, cooling, radiation, or physical viscosity. Start with the 2D problem. The
executable also supports a vertically uniform, periodic 3D slab; this is not a
vertically stratified disk.

## Equilibrium and fixed gravity

In code units, let \(GM=1\), \(r=\sqrt{x^2+y^2}\), and \(a=0.25\). The prescribed,
time-independent potential is

\[
\Phi(r)=\begin{cases}
\displaystyle\frac{GM}{2a}\left(\frac{r^2}{a^2}-3\right),&r<a,\\
\displaystyle-\frac{GM}{r},&r\ge a.
\end{cases}
\]

Both the potential and acceleration are continuous at the core boundary. Set

\[
\Omega^2=\frac{GM}{\max(r,a)^3},\qquad
\mathbf v=(-\Omega y,\Omega x,0),\qquad
\rho=0.01+\exp\left[-\left(\frac{r^2-1}{0.3}\right)^2\right],
\qquad P=10^{-3},\quad\gamma=5/3.
\]

The annulus peaks at radius 1, where the orbital period is \(2\pi\) and the Mach
number is about 25. The low-density background rotates with the same law. The
constant pressure avoids a pressure-gradient correction to the Keplerian speed;
the smooth density profile avoids an initially discontinuous contact. The
regularized core lies well inside the dense ring.

This is an exact stationary solution of the continuum Euler equations:
\(\nabla\cdot\mathbf v=0\), advection is tangent to density and entropy contours,
and \(v_\phi^2/r=\partial_r\Phi\). It is not an exactly preserved discrete
equilibrium. Initial conserved variables are sampled at cell centers, not
integrated over cell volumes. Cartesian reconstruction, source splitting, and
the Riemann solver introduce errors even with artificial viscosity disabled.

Gravity uses the existing problem `addStrangSplitSources` hook. Each kick adds
\(\Delta\mathbf m=-\Delta t\rho\Omega^2(x,y,0)\) and changes gas total energy
by the exact kinetic-energy change at fixed density. It leaves auxiliary
internal energy unchanged. The kick has zero torque about the origin. Hydro
transport plus these kicks is second-order Strang splitting; this does not
guarantee discrete conservation of gas plus potential energy.

The x/y ghost cells are reset to the analytic steady solution. Thus the square
domain has fixed equilibrium boundary data, not periodic seams through the
rotating flow or a closed angular-momentum budget. The dense ring is well away
from those boundaries. Small changes in whole-domain mass, energy, or angular
momentum must be interpreted with possible boundary transport in mind.

## Run and diagnostics

From the repository root:

```sh
./scripts/bash/quokka config -d 2d -DQUOKKA_PYTHON=OFF
./scripts/bash/quokka build -d 2d KeplerianDisk
./scripts/bash/quokka run -d 2d --filter '^KeplerianDisk$'
```

CTest uses a \(64^2\) grid and evolves to \(t=0.25\). It requires positive density
and thermal pressure, finite diagnostics, relative density L1 error below 0.03,
and mass-weighted radial-velocity RMS below 0.03 (the circular speed at radius 1
is unity). This is a short equilibrium regression, not a long-term disk survival
criterion. `disk.test_tolerance` enables those final error bounds; its default
is negative, disabling the bounds for exploratory long runs.

The default input runs one orbit at \(128^2\). For isolated output directories:

```sh
mkdir -p build/2d/keplerian-runs
cd build/2d/keplerian-runs
../src/problems/KeplerianDisk/KeplerianDisk ../../../inputs/KeplerianDisk.toml \
  'amr.n_cell=128 128' hydro.artificial_viscosity_coefficient=0 \
  disk.history_file=n128-k0.txt disk.transport_file=n128-k0-transport.txt > n128-k0.log 2>&1
../src/problems/KeplerianDisk/KeplerianDisk ../../../inputs/KeplerianDisk.toml \
  'amr.n_cell=128 128' hydro.artificial_viscosity_coefficient=0.1 \
  disk.history_file=n128-k01.txt disk.transport_file=n128-k01-transport.txt > n128-k01.log 2>&1
```

Repeat at 256 cells per direction for a resolution comparison. Use separate
working directories if enabling plotfiles (`plotfile_interval=100`), since
other simulation outputs can share filenames. All runs here use uniform grids;
the driver rejects AMR to keep its diagnostic sums unambiguous.

`disk.history_file` is a whitespace-delimited history written at initialization
and after every timestep. It records time, total mass, total \(L_z\), kinetic
energy, gas plus potential energy, relative density L1 error against the initial
analytic profile, radial-velocity RMS, and mass and \(L_z\) inside radius 1.
The inner-region quantities reveal redistribution that a nearly constant total
angular momentum could hide. Their changes include physical flux through
\(r=1\) as the numerical solution departs from equilibrium.

### Final radial transport profile

At the end of evolution, `disk.transport_file` (default
`keplerian_disk_transport.txt`) records the final time and columns
`radius r_lower r_upper mass_flux_outward Lz_flux_outward sampled_volume_over_annulus_volume`.
Positive flux means outward transport of mass or positive z angular momentum;
the conventional positive-inward accretion rate is minus the reported mass flux.
These are instantaneous rates, not time-integrated transport.

For each annular bin, the code averages the **products**
\(\rho v_r=(x m_x+y m_y)/r\) and
\(\rho v_r\ell_z\), where \(\ell_z=(x m_y-y m_x)/\rho\), over valid cell centers.
It then reports

\[
\dot M_{\rm out}(r)=2\pi r\langle\rho v_r\rangle,\qquad
\dot L_{z,\rm out}(r)=2\pi r\langle\rho v_r\ell_z\rangle.
\]

The averages are volume weighted over the finite-width annulus, approximating
the azimuthal mean at its midpoint. In 2D, rates are per unit vertical length;
in 3D they are additionally multiplied by the slab height. They are **advective
fluxes computed from the evolved state**, not the total numerical flux including
artificial viscosity or Riemann-solver dissipation. Isotropic pressure exerts no
torque across a circular surface, and the prescribed central gravity has zero
torque. Numerical stress transport still requires a separate face-flux budget.

`disk.transport_rmax` defaults to the largest complete circle centered on the
origin inside the x/y domain. Larger radii are rejected rather than using partial
annuli. `disk.transport_nbins` defaults to approximately two cells per radial bin;
it can be overridden, but bins narrower than the larger x/y cell width are
rejected. Cells exactly at the origin are omitted, where radial velocity is
undefined. The final column compares sampled cell volume with analytic annular
volume, exposing Cartesian sampling error, especially near the origin. The mean
is normalized by sampled volume, so constant flux-density fields are reproduced
without a cell-count-dependent bias.

CTest enables `disk.check_transport=1`, which verifies the same diagnostic on
manufactured inward and outward flows with constant density, radial speed, and
specific angular momentum. This checks the sign, circumference factor, and
angular-momentum weighting independently of the evolved disk.

Plot one or several profiles with the supplied Matplotlib helper (from the repo
root; the paths below match the isolated runs above):

```sh
python3 src/problems/KeplerianDisk/plot_transport.py \
  build/2d/keplerian-runs/n128-k0-transport.txt \
  build/2d/keplerian-runs/n128-k01-transport.txt \
  --labels 'AV off' 'AV K=0.1' \
  --output build/2d/keplerian-runs/transport.png
```

The plot preserves the signs of both fluxes and labels each curve with its
snapshot time. Supply a `.pdf` output path for vector output. On restricted
macOS environments, prefix the command with
`MPLCONFIGDIR=/private/tmp/quokka-mpl` to use a writable Matplotlib cache.

## What the existing viscosity does

The added face flux in `HydroSystem::ComputeFluxes` is

\[
F_{\rm AV}=A(U_L-U_R),\qquad A=K\max(-D,0).
\]

For an x-face on a square mesh of spacing \(h\), the current estimator is

\[
D_x=\Delta_x u+\tfrac12(dv_L+dv_R),\qquad
dv=\min(v_{j+1}-v_j,v_j-v_{j-1}).
\]

It has velocity units. Taylor expansion of each transverse difference gives
\(dv=h\,v_y-\tfrac12h^2|v_{yy}|+O(h^3)\), and hence in a smooth solenoidal flow

\[
D_x=-\tfrac12h^2|v_{yy}|+O(h^3).
\]

The minimum creates a compression bias even though the true divergence is zero.
Outside the core, \(v=\sqrt{GM}\,x r^{-3/2}\), so

\[
v_{yy}=\sqrt{GM}\,x r^{-7/2}
\left[-\tfrac32+\tfrac{21}{4}\frac{y^2}{r^2}\right].
\]

Thus \(A=O(Kh^2\sqrt{GM}\,r^{-5/2})\), with an angular pattern tied to the mesh.
The y-face result follows by rotation. The leading term vanishes at special
angles; higher-order stencil errors still matter there. This is a bias in the
**coefficient**, not a measurement of the full flux: \(U_L-U_R\) is the jump
between reconstructed states and can already be very small in a smooth region.
It is therefore inappropriate to identify \(A\) directly with a physical
kinematic viscosity or infer an accretion time from it alone.

## Expected effect of a Balsara limiter

A proposed face factor is

\[
f_B=\frac{|\theta|}{|\theta|+|\boldsymbol\omega|+\epsilon c_s/h},\qquad
\theta=\nabla\cdot\mathbf v,\quad\boldsymbol\omega=\nabla\times\mathbf v,
\qquad A_B=f_B A.
\]

This is a finite-volume adaptation of the shear limiter discussed by
[Cullen & Dehnen (2010), section 2.2](https://academic.oup.com/mnras/article/408/2/669/1024183).
It is **not yet implemented in the evolution solver**. The prediction here uses
normal derivatives across the face and transverse centered derivatives averaged
over its two neighboring cells, all calculated from cell-centered velocities.
The divergence and curl must use consistent physical units and directional grid
spacings. Reusing the current biased \(D/h\) in the numerator would instead
leave a first-order compression error.

For exact Keplerian rotation, \(\theta=0\) while
\(\omega_z=\Omega/2\). Thus the continuum limiter is zero. Centered discrete
derivatives leave \(\theta=O(\Omega(h/r)^2)\), giving
\(f_B=O((h/r)^2)\) when vorticity dominates the regularizer. In this regime
\(A_B=O(h^4)\) rather than \(A=O(h^2)\) at fixed radius. This predicts much less
extra momentum/scalar diffusion in smooth rotating regions, not removal of all
numerical angular-momentum transport. At a curl-free compressive shock the
factor approaches unity; shock-plus-shear flows need separate validation.

The source-split intermediate states are not exactly circular. For example,
outside the core, \(\nabla\cdot\mathbf g=GM/r^3\) in 2D, so the first gravity
half-kick introduces an expansive divergence \((\Delta t/2)GM/r^3\). Consequently
the analytic initial-state factors below cannot be assumed to hold at every
Riemann solve. Evolved numerical compressions and the core transition also need
to be measured in an actual limited run.

Run the independent stencil calculation with:

```sh
python3 src/problems/KeplerianDisk/analyze_viscosity.py
```

For x-faces with \(0.7<r<1.3\), \(K=0.1\), and \(\epsilon=10^{-4}\):

| Grid | Fraction with current coefficient active | Mean current coefficient | Sum of limited coefficients / sum of current coefficients |
| --- | --- | --- | --- |
| 64² | 1.000 | 1.74755e-4 | 1.73620e-3 |
| 128² | 1.000 | 4.33221e-5 | 4.29178e-4 |
| 256² | 1.000 | 1.08881e-5 | 1.07609e-4 |
| 512² | 1.000 | 2.71917e-6 | 2.69303e-5 |

These are unweighted face statistics of analytic initial data; y-face statistics
are identical by symmetry. They are not evolved-flow measurements, heating
rates, or measured improvements in disk survival.

## Local validation

The 2D CPU executable builds and the short CTest passes. Full-orbit comparisons
with current viscosity enabled/disabled are recorded below. The Balsara-limited
column above remains a stencil prediction, not a completed simulation comparison.

| Grid | K | Density relative L1 after one orbit | Radial-velocity RMS |
| --- | --- | --- | --- |
| 128² | 0 | 0.119601 | 0.00518602 |
| 128² | 0.1 | 0.119821 | 0.00513897 |
| 256² | 0 | 0.0457508 | 0.00212111 |
| 256² | 0.1 | 0.0455835 | 0.00210666 |

At both resolutions, disabling the explicit viscosity makes little difference
to the density error; the small difference even changes sign. This is evidence
that other discretization errors dominate this particular one-orbit test.
A large reduction in the explicit coefficient
therefore need not yield a comparably large improvement in the evolved disk.
Compare the future limited run against **both** controls, including radial
redistribution and longer evolution, before attributing changes to the switch.

Validation used AppleClang on a CPU with one MPI rank, PPM reconstruction, CFL
0.3, and dual energy disabled. All four runs reached \(t=2\pi\). Histories and
logs are generated under `build/2d/keplerian-runs/`; they are not regression
baselines committed to the repository. The strict-tolerance negative control
(`stop_time=0.02 disk.test_tolerance=0` at 64²) returned failure as expected.
GPU, multi-rank MPI, and 3D execution have not been validated.

### xPPM convergence comparison

Repeating the same 2D one-orbit runs with `hydro.reconstruction_order=5`
(extrema-preserving PPM), while retaining CFL 0.3, disabled dual energy, and
all other input settings, gives:

| Grid | K | Density relative L1 after one orbit | Radial-velocity RMS |
| --- | --- | --- | --- |
| 128² | 0 | 0.054885007873 | 0.00361068245 |
| 128² | 0.1 | 0.054884694598 | 0.00360858708 |
| 256² | 0 | 0.007414701245 | 0.000519560186 |
| 256² | 0.1 | 0.007414255832 | 0.000519457710 |

All four runs completed successfully at exactly \(t=2\pi\). Doubling resolution
reduces the density error by 7.4022 without AV and 7.4026 with AV, corresponding
to observed two-grid orders \(p=\log_2(E_{128}/E_{256})\) of 2.8880 in both cases.
This is a measured rate over these two resolutions, not a claim of asymptotic
third-order accuracy. Artificial viscosity has a negligible effect on these
errors. Compared with the PPM runs, xPPM reduces the density error by about 2.2
times at 128² and 6.2 times at 256².

To reproduce, add `hydro.reconstruction_order=5` to each comparison command
above. These runs use output prefixes `xppm-n128-k0`, `xppm-n128-k01`,
`xppm-n256-k0`, and `xppm-n256-k01` under `build/2d/keplerian-runs/`, with `.log`,
`-history.txt`, and `-transport.txt` suffixes for logs and diagnostics.

### PLM convergence comparison

The same 2D one-orbit comparison with `hydro.reconstruction_order=2` and
`hydro.plm_limiter=sweby` (the default PLM limiter), retaining all other settings,
gives:

| Grid | K | Density relative L1 after one orbit | Radial-velocity RMS |
| --- | --- | --- | --- |
| 128² | 0 | 0.203976259148 | 0.00380681030 |
| 128² | 0.1 | 0.204044834837 | 0.00380638758 |
| 256² | 0 | 0.046399265419 | 0.000886775626 |
| 256² | 0.1 | 0.046403242907 | 0.000886830136 |

All four runs completed successfully at exactly \(t=2\pi\). The error reductions
from 128² to 256² are 4.3961 without AV and 4.3972 with AV, giving observed
two-grid orders of 2.1362 and 2.1366, respectively. AV slightly increases the
density error in both PLM runs, but its effect remains small. PLM has higher
density errors than xPPM at both resolutions; at 256² its density error is close
to that of ordinary PPM.

The output prefixes are `plm-n128-k0`, `plm-n128-k01`, `plm-n256-k0`, and
`plm-n256-k01` under `build/2d/keplerian-runs/`, with `.log`, `-history.txt`, and
`-transport.txt` suffixes. To reproduce, use the comparison commands above with
`hydro.reconstruction_order=2 hydro.plm_limiter=sweby` and distinct output names.

## One-step equilibrium residuals

See [the residual measurements](keplerian_disk_residuals.md) for the spatial
flux/source imbalance, timestep sweep, azimuthal decomposition, and numerical
angular-momentum budget. Reproduce with `measure_residuals.py` in the problem
directory.

The report also compares exact face-center states against Gauss-integrated
analytic face fluxes and gravity. Run `compare_analytic_residuals.py` in the
problem directory to reproduce those controls.
