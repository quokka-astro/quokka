# Keplerian ring spreading

`KeplerianDisk` implements the **hydro-only, fixed-central-mass variant** of the
ring experiment in Krumholz, McKee & Klein (2004), ApJ 611, 399, section 3.4.2
([local paper](../../../Krumholz_2004_ApJ_611_399.pdf)). It measures spreading of an
initially narrow ring on a Cartesian mesh. There is no gas self-gravity, accretion,
particle motion, cooling, MHD, or explicitly prescribed physical viscosity.

## Paper parameters and units

The ring has a full radial width of **two cells**, temperature **10 K**, mean
molecular mass **2.33 proton masses**, and a central mass of **one solar mass**.
`disk.ring_radius_cells` selects the paper's radii **20, 40, or 60** (default 20).
The EOS is isothermal, with physical sound speed
\(c_s=\sqrt{k_B(10\,\mathrm{K})/(2.33m_p)}\).

One code length is one cell, taken to be **1.85 AU**. The paper prints
“2.1 × 10¹³ cm = 1.85 AU”; these are inconsistent. We use its AU value
(2.76756 × 10¹³ cm), which gives \(r_B/\Delta x\simeq1.35\times10^4\),
consistent with its independently quoted \(1.36\times10^4\) to rounding.
The code time unit is \(\sqrt{\Delta x^3/(GM_\odot)}\), so \(GM=1\).
The driver prints its actual Bondi radius and orbital period at startup.

With \(R=\sqrt{x^2+y^2}\) and \(R_0=20,40,60\), initial conditions are

\[
\rho=\begin{cases}1,&|R-R_0|<1,\\10^{-6},&\text{otherwise},\end{cases}
\qquad P=c_s^2\rho,\qquad
\mathbf v=R^{-3/2}(-y,x,0),\qquad \Phi=-1/R.
\]

Density normalization is arbitrary without self-gravity. A top-hat profile,
cell-center sampling, the background density, and boundary conditions are
implementation choices: section 3.4.2 does not specify them. Both ring and
background rotate. The force is unsoftened; integer domain bounds with unit
cell spacing keep the singular origin off cell centers. Analytic face diagnostics
assign zero force/potential at the origin itself as a quadrature convention.
Gravity uses the problem's Strang-split momentum kicks. Isothermal pressure comes
from density; total and auxiliary energy are not used as thermal diagnostics.

The default mesh is 256² on [-128,128]², with analytic fixed x/y boundary data.
The optional 3D build is still a uniform periodic slab of height 8, with cylindrical
gravity and no vertical stratification. Use **2D** for the surface-density ring
experiment; the slab is not a reproduction of the paper's thin 3D disk.
AMR is disabled. Changing mesh size requires changing domain bounds too, preserving
unit x/y cell spacing. The ring must fit entirely inside the domain.

## Run and validation

```sh
./scripts/bash/quokka build -d 2d KeplerianDisk
cmake --build build/2d --target KeplerianDiskAlphaFit
./scripts/bash/quokka run -d 2d --filter '^KeplerianDisk'
# Run the other radii from separate output directories:
build/2d/src/problems/KeplerianDisk/KeplerianDisk inputs/KeplerianDisk.toml \
  disk.ring_radius_cells=40 disk.orbits=0.9
```

`disk.orbits` defaults to 0.9, covering the paper's 0.09–0.9 orbit averaging
interval. An explicit `stop_time` overrides it in code time units. Plotfiles can
be enabled for surface-density fits; the paper fits its equation (23) and converts
the fitted viscosity using \(\alpha=3\nu\Omega/(2c_s^2)\).
No physical viscosity is imposed by the driver. The in-situ diagnostic below
measures a fitted alpha without changing evolution.

CTest checks continuum annulus mass, radial second moment, and angular momentum
against initialized grid sums for all three radii. Exact lattice counts are
272, 492, and 724 cells, respectively; continuum mass/angular-momentum checks allow
10% for Cartesian sampling of sharp edges (the 20-cell ring differs by 8.2%). It also runs a short evolution, checks positive density and
isothermal pressure, and checks transport signs using manufactured inflow/outflow.
The existing one-step residual export remains available via `disk.residual_prefix`.
The earlier smooth-annulus analysis scripts contain that old profile and geometry;
they must not be used as analytic references for this ring.

`disk.history_file` records time, mass, angular momentum, kinetic energy, gas plus
potential energy, relative density change, radial velocity RMS, and mass and
angular momentum inside **R0**. Gas plus potential energy is not conserved under
an isothermal closure. The density change is a spreading diagnostic, not an
equilibrium error: finite pressure also spreads the initial ring. Optional
`disk.test_tolerance` retains the old density-change/radial-speed bounds but is
disabled by default and is not a long-term ring-survival criterion.

`disk.transport_file` retains its six-column instantaneous advective-flux format
and compatibility with `plot_transport.py`. Positive rates are outward. These
state-derived rates exclude numerical stress fluxes. `disk.transport_rmax` and
`disk.transport_nbins` select complete annular bins at least one cell wide.
In 2D the density is interpreted as surface density; 3D rates include slab height.

The earlier equilibrium derivation and viscosity discussion are preserved in
[the historical smooth-annulus notes](keplerian_disk_smooth_annulus.md). The current
setup matches the ring parameters above, subject to the documented cell-size
ambiguity; it does not reproduce the paper's sink algorithm or measured viscosity.

## In-situ effective alpha

The diagnostic fits equation (23) at positive times, with initial radius and
**initial ring mass held fixed**, and converts the fitted width parameter using

\[
\Sigma(x,\tau)=\frac{m}{\pi R_0^2}\tau^{-1}x^{-1/4}
\exp\!\left[-\frac{1+x^2}{\tau}\right]I_{1/4}(2x/\tau),
\quad x=R/R_0,\qquad
\nu=\frac{\tau R_0^2}{12t},\qquad
\alpha=\frac{3\nu\Omega(R_0)}{2c_s^2}.
\]

`disk.measure_alpha=true` enables measurements by default. Samples are separated
by at least `disk.alpha_interval_orbits=0.01`, evaluated at actual timestep times;
the first positive time and final time are also sampled. No fit or division by
time is attempted at initialization. `disk.alpha_file` defaults to
`keplerian_disk_alpha.txt`, with columns

```text
time orbits tau nu alpha fit_relative_L2 status alpha_mean covered_orbits
```

Time and viscosity use the simulation's code units; alpha is dimensionless.
The header records the initial mass, radius, sound speed squared, orbital period,
and fit-bin geometry. The density profile subtracts the specified initial
background without clipping negative excesses. The initial excess mass is
computed from the sampled **initial analytic state**, not refitted or replaced
by the current mass. Only valid cells contribute; all MPI ranks are reduced.
In 3D, density is averaged through the uniform slab and multiplied by slab height,
with the same height included in the initial mass.

Fits use all complete annuli from the origin to the inscribed domain radius,
with bins approximately one cell wide. Each measured surface density is the
mean over sampled cells. The analytic model is area-averaged over each bin with
eight-point Gauss quadrature. The fit minimizes the **unweighted sum of squared
surface-density residuals** over bins; the paper does not specify fitting weights
or radial binning. Its mass normalization is fixed, leaving only tau to fit.
A logarithmic scan and bracketed minimization search `1e-5 <= tau <= 10`.
A scaled Bessel evaluation avoids exponential overflow for narrow rings.

Status 0 denotes an interior minimum, 1 a search-bound minimum, 2 nonfinite data,
and 3 no positive signal. Invalid fits report `nan` for nu and alpha and are
excluded from the average. `fit_relative_L2` is the residual norm divided by the
data norm; status 0 does **not** by itself indicate that the ring resembles the
analytic model closely.

`alpha_mean` is a trapezoidal **time-weighted** average, with linear interpolation
at the boundaries of the paper's 0.09–0.9 orbital-period interval. Invalid samples
break interpolation, leaving gaps. `covered_orbits` records the actual valid
coverage: **0.81 is required for the full paper interval**. A short run reports a
partial average, or `nan` before any interval has been covered. The final average
and coverage are also printed. Restarted runs start a new diagnostic history and
average for that invocation; prior samples are not restored from checkpoints.

This is the paper's effective-width method, not a stress-based or instantaneous
viscosity measurement. It uses elapsed time, not the time derivative of fitted
tau. The finite two-cell initial width is not subtracted, so very early alpha
values are biased high. Pressure-driven spreading, Cartesian sampling, radial
variation in numerical diffusion, and the absence of an accreting sink can also
produce departures from the pressureless constant-viscosity model. Inspect the
fit residual along with alpha; the diagnostic does not isolate physical shear
viscosity from these effects.

The `KeplerianDiskAlphaFit` CTest recovers known widths from independent
high-precision Bessel/integral reference profiles, including cold/narrow rings,
and checks radius/slab rescaling, unit conversion, fit failures, and averaging
with clipped boundaries and missing samples. `disk.check_alpha=1` additionally
exercises Cartesian binning and MPI reduction using a manufactured ring with
known tau; the three ring-radius CTests enable this check.

## Local validation

A single-rank 2D CPU build passed the ring setup and residual CTests. Running the
default input with `disk.check_initial=1` completed 0.9 orbit (1966 steps,
code time 505.7866606). Relative mass change was -7.66e-11; relative angular
momentum change was -0.013. Final density relative L1 change was
1.50653 and radial velocity RMS was 0.0150056 in code units. These characterize
this numerical run, not agreement with the paper's measured alpha. GPU and 3D execution were not validated.

With alpha measurements enabled, all five `^KeplerianDisk` CTests passed. The
default 0.9-orbit run produced 89 valid fits, a time-averaged alpha of
1.5915905 over the full 0.81-orbit averaging interval, and final alpha
0.81382386. The final relative profile-fit residual was 0.272996; the analytic
profile is an imperfect description of the evolved ring. Independent trapezoidal
integration of the saved samples reproduced the in-situ average.

A two-rank CPU run also passed the manufactured-profile and initialization checks:

```sh
# From build/2d/keplerian-alpha-mpi-validation:
mpirun --oversubscribe -np 2 ../src/problems/KeplerianDisk/KeplerianDisk \
  ../../../inputs/KeplerianDisk.toml max_timesteps=1 \
  disk.check_initial=1 disk.check_alpha=1
```

The manufactured Cartesian profile recovered tau = 0.050591 for an imposed 0.05
(the difference reflects annular Cartesian sampling). MPI required running outside
the macOS sandbox. GPU and 3D measurements remain unvalidated.

## Figure 10 comparison with standard PPM

The three-radius comparison uses `hydro.reconstruction_order=3` (standard PPM),
artificial-viscosity coefficient 0.1, CFL 0.3, and density floor **1e-7**. The initial
background density remains 1e-6. All three runs used the same 2D executable and
256² mesh, and completed 0.9 orbit with full 0.09–0.9 averaging coverage.

| R/Δx | Time-averaged alpha | Final alpha | Final relative profile-fit L2 |
|---:|---:|---:|---:|
| 20 | 1.6806789 | 0.76764707 | 0.288226 |
| 40 | 0.42878381 | 0.15607513 | 0.357784 |
| 60 | 0.32911837 | 0.15575656 | 0.261443 |

The descriptive log-space fit through these three measurements is
\(\alpha\simeq154.66(R/\Delta x)^{-1.53657}\). The reference curves in the remade
figure use the paper's equations (22) and (24), with its quoted Bondi resolution
1.36e4; the ring reference is the published power law, not digitized figure points.
The companion time-history plot shows substantial variations and imperfect
profile fits: the 60-cell case reaches a relative L2 residual of 0.633 within the
averaging window. The fitted alpha should be interpreted as the effective-width
diagnostic described above.

The floor was raised for this comparison because the original 1e-10 floor led
to timestep collapse in the larger PPM rings. Short 60-cell trials at 1e-8, 1e-7,
and 1e-6 motivated the 1e-7 choice; it is below the initial background density.
The complete 20-cell PPM result is unchanged from its original-floor run. The
previous xPPM runs at radii 40 and 60 failed early and are not included. No
reconstruction or source-term implementation was changed for this comparison.

Run the campaign in a fresh output directory, then regenerate plots from the
saved histories by omitting `--run`:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 \
  src/problems/KeplerianDisk/reproduce_fig10.py --run \
  --reconstruction-order 3 --density-floor 1e-7 \
  --output build/2d/keplerian-figure10-ppm-floor7
```

The output directory contains `figure10.png`/`.pdf`, `alpha_history.png`/`.pdf`,
`measurements.csv`, `summary.json`, and a README. Each `r20`, `r40`, and `r60`
subdirectory retains its diagnostic histories, log, and exact command.
`provenance.json` records executable/input/source hashes and the numerical
overrides. The plotting script independently integrates every saved alpha
history and rejects incomplete averaging coverage or disagreement with the
in-situ average. `--run` refuses to overwrite existing run directories.

### Restricting the averaging window to 0.1–0.3 orbits

Reintegrating the saved PPM/floor-1e-7 alpha histories over 0.1–0.3 orbits gives
mean alpha values **1.87243157**, **0.61928998**, and **0.58372384** for radii
20, 40, and 60 cells. These use the existing per-snapshot Pringle fits, with
linear interpolation at the interval endpoints and trapezoidal time averaging;
the simulations and per-snapshot fits are unchanged. The histories from the
plotfile-every-10-step reruns are byte-identical to those used here.

The descriptive three-point radial fit becomes
\(\alpha\simeq48.8602(R/\Delta x)^{-1.11858}\). Maximum relative profile-fit L2
residuals within the selected window are 0.1537, 0.1313, and 0.1613.
The paper's equation (24) reference still represents its original 0.09–0.9-orbit
average, and the comparison figure labels that distinction.

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 \
  src/problems/KeplerianDisk/reproduce_fig10.py \
  --data-dir build/2d/keplerian-figure10-ppm-floor7 \
  --output build/2d/keplerian-alpha-ppm-orbits01-03 \
  --average-start 0.1 --average-end 0.3
```

The new output directory preserves the earlier comparison and contains updated
PNG/PDF figures, CSV measurements, summary JSON, and a README. Its time-history
figure shades times outside the selected averaging window.

### Figure 8 density-threshold diagnostic on the ring runs

`src/problems/KeplerianDisk/reproduce_fig8.py` reads the three saved
`build/2d/keplerian-r{20,40,60}-ppm-plots10-20260910` campaigns and writes
`build/2d/keplerian-figure8-ppm/figure8.png` and `.pdf`, per-run CSV histories,
compressed radial profiles, and method/provenance notes. It uses one-cell
annular averages of the actual initial and evolving surface density. The edge
is the first annular center outside four cells satisfying
\(\Sigma(r,t)\geq0.9\Sigma(r,0)\). Background gas is retained; there is no
interpolation or smoothing. The inner-four-cell exclusion follows the paper's
no-accretion convention. Plotfile times and density integrals are checked
against the in-situ histories, and the initial Cartesian densities against
the prescribed ring.

The figure follows the paper's logarithmic axes, paired AU/cell radius scales,
and diamond/plus/star curves (here representing the three ring radii). Time
is in periods at \(4\Delta x\), consistent with equation (18), and radius
uses the runs' adopted \(\Delta x=1.85\) AU. Axis limits adapt to these data.

This is a literal application of the criterion to the section 3.4.2 rings,
not a reproduction of the section 3.4.1 continuous-disk initial conditions
used in Figure 8. In the rings' interiors it measures dilute background gas.
The 20-cell result is always 4.5 cells; the 40- and 60-cell results peak at
6.5 and 8.5 cells, respectively, and all finish at 4.5 cells. These curves
should not be interpreted as the geometrical ring edge or used to infer
the paper's disk-evacuation power law.

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/reproduce_fig8.py
```

### Continuous power-law disk (section 3.4.1)

Use `inputs/KeplerianDiskPowerLaw.toml` for the stationary, non-accreting
continuous disk. `disk.profile = "power_law"` initializes
\(\Sigma=\Sigma_0(r_0/r)\) inside \(r_0=2\times10^{15}\) cm,
with \(\Sigma_0=0.1\) g cm\(^{-2}\), and the existing dilute background
outside. Density is stored in units of \(\Sigma_0\); hence the outer disk
has code density one, the background is `1e-6`, and the floor is `1e-7`.
Gas self-gravity is disabled, so this normalization does not alter the flow.
The fixed solar-mass potential, Keplerian velocities, and 10 K isothermal EOS
are retained. This is the paper's non-accreting variant, not its accreting or
advected sink experiments.

The adopted cell length remains 1.85 AU (rather than the paper's inconsistent
printed cm value); the outer disk radius is approximately 72.27 cells. The
256-square domain, PPM order 3, CFL 0.3, and artificial viscosity 0.1 are
retained. For this profile `disk.orbits` counts periods at **four cells**,
with a default of 50. The new input writes plotfiles every 10 steps.
The ring profile remains the default for the original input, with its original
orbital-period definition. Ring-profile alpha fitting is disabled and rejected
if explicitly enabled on a power-law disk.

`KeplerianDiskPowerLaw` is a two-step CTest that checks the initial mass against
the independent continuum integral and bounds the early density/velocity error.
The Figure 8 analysis also checks the exact initial density in every cell:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/reproduce_fig8.py \
  --power-law-run build/2d/keplerian-powerlaw-ppm-plots10-20260910 \
  --output build/2d/keplerian-powerlaw-figure8
```

The completed 2026-09-10 power-law run reached 50 periods at four cells
(code time 2513.2741228718346) in 11,044 steps, saving 1,106 plotfiles
(initial, every 10 steps, and final). The final threshold radius is 9.5 cells
(17.575 AU); the maximum sampled radius is 10.5 cells. Mass changed by
-0.0362%. All six KeplerianDisk CTests passed, and a 500-step four-rank
check agreed with the single-rank history to within 2.5e-13 in
`abs(delta)/(1+abs(value))`. The production run used one rank.
All plotfile density integrals and times passed comparisons to the in-situ
history; the output cadence and executable hash were verified.

### Sub-cell evacuation-radius estimate

The Figure 8 analysis now locates the first outward zero of
\(F(r)=\Sigma(r,t)-0.9\Sigma(r,0)\) by linear interpolation between
adjacent annular samples with \(F_i<0\leq F_{i+1}\):

\[r_{\rm evac}=r_i-\frac{F_i}{F_{i+1}-F_i}(r_{i+1}-r_i).\]

It interpolates the density difference, using both measured profiles, without
smoothing in time. Samples inside four cells remain excluded. If the first
eligible sample at 4.5 cells already exceeds the threshold, the interface is
unresolved within the inner cutoff: the CSV records NaN and
`inner_edge_unresolved=1`, and the plot leaves a gap. Exact threshold equality
at that sample gives radius 4.5. Sub-cell interpolation does not establish a
sub-cell physical error bound.

The saved 50-period power-law run gives a final radius of 9.4963129857 cells
(17.5681790236 AU). Of 1,106 outputs, 888 have bracketed crossings and 218
have unresolved inner edges. Manufactured linear profiles with varying initial
density check fractional roots, exact equality, first-crossing selection, and
missing crossings. Independently evaluating the interpolated density residual
at every reported root gives a maximum absolute residual of 2.3e-15.

Updated PNG/PDF, CSV, profiles and method notes are in
`build/2d/keplerian-powerlaw-figure8-subcell`; earlier un-interpolated results
remain in `build/2d/keplerian-powerlaw-figure8`. Reproduce with the same command
as above, changing `--output` to the subcell directory.

### Comparison with Equation (18)

`src/problems/KeplerianDisk/fit_evacuation.py` fits all 888 resolved sub-cell
radii over 3.699576–50 periods by unweighted least squares in log space
(one weight per plotfile). The result is
\(r_{\rm evac}/\Delta x=2.935308\,T^{0.292791}\), where
\(T=t/P(4\Delta x)\). The log-space coefficient of determination is 0.8295.
Unresolved radii are excluded. The data are temporally correlated, so no
independent-sample confidence interval is claimed. The coefficient at one
period is an extrapolation outside the fitted interval.

Equation (18) gives \(6.1T^{0.23}\). At 50 periods our fitted radius is
9.228 cells (measured 9.496), compared to the paper's 15.000 cells: the fit
is about 38.5% smaller. Our exponent is 0.063 larger. Equation (18) explicitly
fits a calculation **with accretion**, whereas this run has no accretion;
the retained cell-length convention also differs from the printed cm value.
These are empirical comparisons between different numerical experiments.

The comparison plot, CSV, fit JSON with input hash, and method notes are in
`build/2d/keplerian-powerlaw-equation18`. Reproduce with:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/fit_evacuation.py
```

### Figure 10 with evacuation-based alpha

`src/problems/KeplerianDisk/plot_evacuation_alpha.py` substitutes the saved
evacuation fit into Equation (21) and plots
\(\alpha=1.597837\times10^4(r/\Delta x)^{-2.915411}\) against the paper's
Equations (22) and (24). Marked values at 20, 40, and 60 cells are 2.573328,
0.341090, and 0.104590. These are extrapolations, not measured ring averages;
the fitted evacuation radii cover approximately 4.3–9.2 cells.

The outputs are `build/2d/keplerian-figure10-evacuation-alpha/figure10.png`
and `.pdf`, plus CSV values, source hashes, model parameters, and method notes.
The calculation is independently checked by direct Equation (21) substitution.
Reproduce with:

```sh
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/plot_evacuation_alpha.py
```

### Joseph et al. (2023) resolved inviscid ring

The `KeplerianDiskJoseph` executable in this problem directory implements the
inviscid Cartesian setup of `2308.03881v1.pdf`, sections 2 and 4 and Appendix E.
It has a separate driver because the earlier Krumholz cases use a globally
isothermal EOS, whereas this experiment needs a radial temperature profile.
The older drivers and results remain available.

Use `inputs/KeplerianDiskJoseph.toml`. Units are `G=M_star=R0=1`, with density
normalized to `Sigma_ref=Sigma_ring(tau0,1)`. Initial density is Equation (2)
at `tau0=0.018`, plus a uniform `1e-7 Sigma_ref` background. The domain is
`[-2,2]^2`, initially resolved by `1024^2` cells, with no refinement. There is
no explicit physical viscosity or artificial viscosity. The hydro settings are
HLLC, linear reconstruction with the newly supported van Leer harmonic limiter,
and CFL 0.4. No characteristic reconstruction is enabled.

The aspect ratio is `h=0.005`. Following the paper's Athena++ method in
Appendix E, the internal energy relaxes exponentially toward
`P/rho=h^2/R` on a timescale `0.01/Omega_K`. The adiabatic gamma used between
relaxation updates is 1.4; the paper does not specify it. Dual energy is enabled
for the cold, highly supersonic flow. Also following Appendix E, the floor is
`1e-15 Sigma_ref`, below the background, rather than imposing the background
as the floor. Thermal and density units are dimensionless; no Krumholz AU
or solar-mass density normalization is applied.

The potential is unsoftened `-1/R`. The origin lies between cells and is never
sampled. Initial radial velocity is zero and azimuthal velocity is
`sqrt((1-h^2)/R)`, matching the pressure-corrected Keplerian velocity prescribed
at the paper's polar boundaries. The paper does not fully specify the initial
Cartesian velocity field; this correction accounts for the imposed radial
sound-speed dependence but not the ring's density gradient. No random density
perturbation is added to the inviscid comparison.

Inside `R=0.2`, density relaxes to the background and radial velocity to zero;
azimuthal velocity is preserved by damping. The damping rate has a quadratic
ramp `(1-R/0.2)^2` and a timescale of one orbit at `R=0.2`, selectable through
`disk.damping_periods`. The paper references a damping prescription without
specifying its timescale in this article; this choice is recorded rather than
claimed to reproduce its exact damping operator. Cooling, central damping,
and gravity are applied in source substeps. Outer boundaries use zero-gradient
outflow (`foextrap`), without an extra inflow-clipping rule. These details and
Quokka's source integration differ from the PLUTO implementation used for Fig. 7.

`disk.orbits` defaults to 748 and is measured at `R0=1` (one orbit is `2pi`
code time units). `disk.profile_interval_orbits` defaults to one. Profiles are
written initially, at the first timestep reaching each output interval, and
at the final time, as `joseph_profile_XXXXXXXX.txt`. Each contains actual time,
orbital time, and columns `R`, `slice_y0`, `Sigma_annular`, `initial_Sigma`.
The slice is linearly interpolated between the two rows straddling `y=0`,
using the positive-x half of the disk. Annular averages use one-cell radial
bins and equal cell-area weighting. Only the density is copied to the CPU at
output cadence; profile sums work across MPI ranks. Full plotfiles are disabled
by default to avoid excessive storage; checkpointing is enabled every 10,000
steps. No physical-viscosity comparison runs are included.

```sh
cmake --build build/2d --target KeplerianDiskJoseph -j 8
python3 src/problems/KeplerianDisk/run_joseph.py --output build/2d/joseph-production
MPLCONFIGDIR=/private/tmp/quokka-mpl python3 src/problems/KeplerianDisk/plot_joseph.py \
  build/2d/joseph-production --output build/2d/joseph-production/profiles
```

The run helper preserves the exact command, input copy and source/binary hashes
and refuses to overwrite an existing directory. `--executable` can select an
HPC/GPU build and `--ranks` an MPI rank count. The plotting helper defaults to
224 and 748 orbits (the two inviscid curves in Fig. 7); `--times` selects other
times. It interpolates between saved times and refuses to extrapolate missing
simulation outputs. It saves PNG/PDF, selected CSV profiles, all profiles in
NPZ, and interpolation metadata.

Local validation on 2026-09-10:

- All seven KeplerianDisk CTests passed after rebuilding the affected drivers.
  `KeplerianDiskJoseph` checks analytic initialization and independently checks
  the gravity kick, thermal relaxation, density damping and radial-velocity
  damping, including preservation of azimuthal velocity by damping. It also
  exercises the van Leer limiter and four hydrodynamic timesteps.
- A `128^2` run completed two orbits in 6,347 steps. Its 21 surface-density
  profiles and preview plot are in `build/2d/joseph-128-validation`.
- A `1024^2` benchmark completed 100 steps, reaching 0.0015139894 orbits in
  82.9 seconds on one CPU rank while the low-resolution check was also running.
  Profiles are in `build/2d/joseph-1024-benchmark`. This is a short benchmark,
  not a long-time Figure 7 result. A naive extrapolation of its startup rate
  to 748 orbits is about 474 days; future timestep and hardware changes can
  alter this substantially. The unsoftened central orbit limits the timestep.
- An independent direct Bessel-series evaluation matched both initial slices
  to absolute error below `9e-15`. The full-resolution 224/748-orbit run has
  **not** been completed. Production MPI runs remain untested; CUDA validation is
  recorded below.

The Joseph input now sets `init_shrink=0.01` to reduce the first CFL timestep
by a factor of 100. Subsequent timesteps grow under the normal timestep-growth
limit. `run_joseph.py --init-shrink` overrides this value, and `--verbose`
records timestep diagnostics. This does not change the steady CFL constraint.
`--restart PATH` resumes a checkpoint in a fresh output directory; profiles
from successive segments can be combined for plotting.

The 1024-square startup check with `init_shrink=0.01` completed 100 steps.
The first timestep was 8.1636841e-07, exactly 0.01 times the control's
8.1636841e-05; the final timestep recovered to 9.47547447e-05.
The startup ramp works as intended, but the small ongoing timestep remains.
Results and verification are in `build/2d/joseph-1024-initshrink001`.


H200 deployment on 2026-09-10:

- Host: `dev-amd24-h200`; working directory:
  `/mnt/ffs24/home/wibkingb/quokka-joseph-20260910`.
- Built `KeplerianDiskJoseph` with CUDA 13.1.2, GCC 13.2.0, OpenMPI 4.1.6,
  and CUDA architecture 90. The GPU CTest passed. A 128-square GPU run completed
  two orbits in 6,354 steps; the maximum relative L1 difference from temporally
  interpolated CPU profiles was `4.12e-5` for the y=0 slice and `1.49e-5` for
  annular averages. The CPU comparison used the earlier `init_shrink=1`; the
  GPU run used `0.01`.
- Use `--blocking-factor 1024 --max-grid-size 1024` with `run_joseph.py` for
  this 2D, 1024-square GPU run. A 1,000-step benchmark took 26.7 seconds with
  this layout, versus 81.9 seconds with the original 128-square grids. Final
  y=0 profiles matched exactly; annular profiles agreed to roundoff.
- The detached `run-production.sh` targets 748 orbits in `inviscid-1024`, with
  radial profiles every orbit, checkpoints every 250,000 steps, and no full
  plotfiles. On successful completion it runs `plot_joseph.py` for the 224- and
  748-orbit inviscid curves. This host is a shared development node, and the run
  uses one H200 directly. `environment.sh` records its GPU UUID and sets
  `AMREX_THE_ARENA_INIT_SIZE=1073741824` to start with a 1 GiB device memory pool
  that can grow as required.
- The benchmark reached 0.01382462 orbits. Extrapolating this startup rate gives
  approximately 16.7 days for 748 orbits; evolving timesteps and GPU contention
  can change the runtime. This estimate is not a completed Figure 7 result.

Local deployment records, validation profiles, benchmark logs, CPU/GPU comparison
results and launch instructions are in `build/2d/joseph-h200-transfer/`.
`source_manifest.json` preserves the copied source hashes; `source_amendments.json`
records subsequent launcher-only updates. Each remote run also preserves its
input and exact invocation in `run.json`.


The 748-orbit production run was stopped after the user imposed a one-hour
wall-time budget. A floor study kept 1024-square cells and both AMReX blocking
factor and maximum grid size at 1024. Each case completed 1,000 steps:

| Density floor | Orbital time reached | Final CFL timestep |
|---:|---:|---:|
| 1e-15 | 0.01382462 | 8.539144e-5 |
| 1e-9 | 0.01382462 | 8.539144e-5 |
| 1e-7 | 0.01365393 | 8.535526e-5 |
| 1e-6 | 0.01357516 | 8.524916e-5 |

The limiting cells are adjacent to the origin, with speed about 18.2 and sound
speed 0.113. Higher floors did not increase the timestep. The new short run
therefore retains `density_floor=1e-15` and targets one orbit, with radial profiles
every 0.05 orbit and checkpoints every 20,000 steps. Its directory on the same
host is `inviscid-1024-one-orbit`. `run-short.sh` invokes `run_short.py`, which
plots the initial, midpoint and final profiles at the times actually reached.
These shorter-time profiles are not the paper's 224/748-orbit Figure 7 curves.

`run_joseph.py --max-walltime 00:50:00` uses the existing simulation wall-time
limit, which stops evolution at about 45 minutes to reserve final-output time.
An outer `timeout --kill-after=30s 55m` bounds the entire launch and plotting
workflow below one hour. The run metadata now records `achieved_orbits` and
`target_reached`, including successful stops before the orbital target.
A ten-second wall-time smoke test exited successfully after 12.2 seconds including
startup and output, wrote a final checkpoint and profile, and generated PNG/PDF
plots. Floor-study scripts, results and this smoke test are saved locally in
`build/2d/joseph-h200-floor-study`.

### Two-orbit Joseph viscosity fits

The short 1024-square run was subsequently stopped at the user's request after
19.6 minutes; its last saved profile is at 0.200008 orbit. The requested comparison
now includes only complete two-orbit runs at 128, 256 and 512 cells per side.
The new 256/512 runs use separate H200 GPUs, a single AMReX box covering each
mesh, the unchanged physical setup, and the same native and outer wall-time caps.

`fit_joseph_viscosity.py` fits Equation (2) to each saved y=0 surface-density
slice over `0.2 <= x/R0 <= 1.8`, with fixed initial normalization and background.
The model is sampled at the radii of the two rows straddling y=0, matching the
profile measurement operator. Radial residuals are unweighted. A straight line
with a free intercept is then fitted to all recovered tau values over 0--2 orbits.
With `T=t/P0`, the conversion is `nu=(d tau/dT)/(24*pi)` and
`alpha(R0)=nu/0.005^2`. These are transient effective viscosities; Appendix D
instead uses a long late-time interval of approximately 180 orbits.

The script requires NumPy, SciPy and Matplotlib and refuses unsuccessful,
incomplete or duplicate-resolution runs. It exports PNG/PDF, the recovered tau
history and interval viscosities, a measurements CSV, fit residuals and hashes.
Its synthetic check independently constructs profiles with unscaled Bessel values
and recovers a known viscosity to relative error below 1e-6. The 128-square CPU
and GPU viscosity fits agree within 2.3e-6 relative.

```sh
python3 src/problems/KeplerianDisk/fit_joseph_viscosity.py \
  build/2d/joseph-h200-transfer/gpu-128-validation-small-arena \
  build/2d/joseph-h200-transfer/inviscid-256-two-orbits \
  build/2d/joseph-h200-transfer/inviscid-512-two-orbits \
  --output build/2d/joseph-two-orbit-viscosity
```


The 256- and 512-square runs both completed two orbits successfully on
2026-09-10, in 699.8 and 750.5 seconds respectively, within their one-hour caps.
The combined figure is `build/2d/joseph-two-orbit-viscosity/viscosity_fits.png`
(and PDF), with a matching `measurements.csv` and per-resolution tau histories.
The full 0--2 orbit fits give:

| Cells per side | nu in code units | alpha at R0, h=0.005 |
|---:|---:|---:|
| 128 | 1.6640799211e-4 | 6.656319684 |
| 256 | 3.9676507476e-5 | 1.587060299 |
| 512 | 6.9315531929e-6 | 0.277262128 |

Maximum relative L2 profile residuals are 9.3%, 7.1% and 3.3%, respectively.
Fits restricted to 1--2 orbits give nu values of 1.6566e-4, 3.9930e-5 and
8.0169e-6, illustrating remaining transient sensitivity, particularly at 512.
The independently executed local and remote post-processing agreed to better
than 1e-9 relative in the recovered viscosities. No incomplete runs or shorter
benchmarks are included in the figure.

Adding `--power-law` fits the three positive viscosity measurements using
unweighted least squares in log space. The two-orbit data give
`nu = 3.576948816e-5 (N/256)^(-2.292701126)`, with log-space R-squared
`0.996818824`. Equivalently, `nu = 11.88193592 N^(-2.292701126)` or
`nu proportional to dx^(2.292701126)` at fixed domain size. The fit is descriptive
of these three early-time measurements. Updated PNG/PDF and `power_law.json` are
in `build/2d/joseph-two-orbit-viscosity-powerlaw`.

`--paper-figure8` overlays the nine Cartesian PLUTO and Athena++ measurements
and their quoted one-standard-deviation errors from Table 1, which underlie
Figure 8 of `2308.03881v1.pdf`. The comparison uses viscosity directly in
`R0^2 Omega0 = sqrt(G M_star R0)` units; it does not mix the paper's secondary
alpha axis at h=0.05 with the h=0.005 alpha values in our measurement CSV.
Paper trend lines are unweighted log-space refits to the rounded tabulated
values (exponents -2.09088 for PLUTO and -2.03834 for Athena++), and are labeled
as table fits. Each curve is drawn only over its measured resolution range.

Combined and standalone PNG/PDF plots, the source values in `paper_figure8.csv`,
and provenance metadata are in `build/2d/joseph-two-orbit-figure8-comparison`.
Our three viscosity values and power-law parameters are unchanged. The figure
marks our 0--2 orbit interval and the paper's longer late-time measurements.

### Ten-orbit extension

The 128/256/512 campaign is extended to ten orbits with unchanged physics and
numerical settings. The 128-square case starts again at t=0 because its original
validation did not save checkpoints. The 256/512 cases restart their final
checkpoints at two orbits. Their original profiles are merged into the new run
only after checking the overlapping restart profile and executable hash. A
single-step 256-square restart check matched the prior final profile and advanced
past two orbits successfully.

New remote directories are `inviscid-128-ten-orbits`, `inviscid-256-ten-orbits`
and `inviscid-512-ten-orbits` under the existing H200 working directory. Each
initial segment has a native `max_walltime=00:59:00` (evolution stops at about
54 minutes if necessary), and an outer 59-minute timeout with a 30-second kill
grace. The subsequently authorized budget is two hours per case from the
original start at 2026-09-10T19:50:15Z. If the initial segment stops before ten
orbits, `extend_and_finish_ten.py` resumes its final checkpoint into a separate
`inviscid-N-ten-orbits-continued` directory with the remaining budget. It merges
the profile history after checking the overlapping restart profile and binary
hash. Profiles remain spaced by
0.05 orbit. The three cases use separate GPUs; no 1024-square run is launched.

`fit_joseph_viscosity.py --end-orbits 10 --power-law --paper-figure8` requires
complete profiles spanning 0--10 orbits before performing each fit. It fits
viscosity over the full interval and reports `nu_second_half` over 5--10 orbits
as a check of transient sensitivity. Omitting `--end-orbits` retains the earlier
two-orbit interval and reproduces the previous viscosity values. The original
two-orbit plot artifacts remain available. Campaign launch scripts and restart
provenance are recorded in `build/2d/joseph-ten-orbit-campaign`.
`collect_ten.py` waits for the remote completion record, downloads the profile
histories without the large checkpoints, and reproduces the final fit and plots
locally in `build/2d/joseph-ten-orbit-viscosity`. Its completion record lists any
case that did not reach ten orbits; such a case is excluded from the ten-orbit fit.

The completed 128/256 cases can also be fit using only 5--10 orbit profiles:

```sh
python3 src/problems/KeplerianDisk/fit_joseph_viscosity.py \
  build/2d/joseph-ten-orbit-campaign/inviscid-128-ten-orbits \
  build/2d/joseph-ten-orbit-campaign/inviscid-256-ten-orbits \
  --start-orbits 5 --end-orbits 10 --power-law --paper-figure8 \
  --output build/2d/joseph-five-to-ten-orbit-viscosity-128-256
```

This selects 101 profiles per resolution and gives viscosities
`7.59915673597e-5` and `2.21152938485e-5`, respectively. The power law is
`nu = 2.21152938485e-5 (N/256)^(-1.78079492268)`. With only two resolutions,
the power law interpolates the measurements and provides no independent test
of the scaling. Its line is restricted to 128--256. These viscosities reproduce
the previous 0--10 fit's second-half diagnostics; the default full-interval fits
remain unchanged. When selecting a new interval, `nu_second_half` refers to
the second half of that interval (7.5--10 here). All plotted samples, exported
tau histories, and fit-profile hashes use the selected interval.

All three ten-orbit runs subsequently completed successfully within the extended
budget: approximately 21.6, 22.6, and 77.3 minutes from campaign launch for
128/256/512. The 512 case resumed at 6.633831573 orbits after its first segment
reached the native wall-time limit. The continued directory contains the checked,
merged profile history; all three cases use the same executable hash.

Adding `build/2d/joseph-ten-orbit-campaign/inviscid-512-ten-orbits-continued`
to the command above and using output directory
`build/2d/joseph-five-to-ten-orbit-viscosity` gives the final three-resolution
5--10 orbit comparison. The 512 viscosity is `5.24280266230e-6`, and the combined
power law is `nu = 2.06541444111e-5 (N/256)^(-1.92871459249)` with log-space
R-squared `0.99804321`. The 128/256 values remain unchanged. Each viscosity
matches the independent second-half diagnostic from the completed 0--10 analysis
to relative tolerance `1e-12`. The selected 512 history contains 102 profiles,
including one extra output at the restart time; the other cases each contain 101.
Both full-duration and selected-window plots and CSVs are preserved separately.

### PPM and xPPM controls at 128 squared

The next campaign runs fresh 128-square initial conditions sequentially with
`hydro.reconstruction_order=3` (PPM), then `5` (extrema-preserving xPPM).
`run_joseph.py --reconstruction-order` records the override in the executed
command. The existing PLM runs used order 2 with the van Leer limiter.
Both controls retain the existing GPU executable, CFL 0.4, initial timestep
factor 0.01, density floor `1e-15`, and 0.05-orbit profile cadence. Each targets
ten orbits with its own two-hour budget (native limit 119 minutes, outer timeout
119 minutes plus 30-second kill grace). Checkpoints are saved every 20000 steps.

The sequential workflow launched at 2026-09-10T22:25:45Z on GPU 0 of
`dev-amd24-h200`. The remote directories under the existing campaign root are
`inviscid-128-ppm-ten-orbits` and `inviscid-128-xppm-ten-orbits`. Each successful
ten-orbit case is automatically fit over 5--10 orbits before proceeding to the
next case. An incomplete case is recorded as such and is not fit over ten orbits.
Launch scripts, collection records, profiles, and fit artifacts are retained
locally under `build/2d/joseph-reconstruction-campaign`; the remote status file
is `reconstruction-128-status.json`.

Both controls completed ten orbits successfully, in the requested order. Their
launcher elapsed times were 45.7 seconds (PPM) and 42.4 seconds (xPPM). Over
5--10 orbits the nominal viscosities are `2.94802925180e-4` (PPM) and
`1.01481498469e-4` (xPPM), compared with the existing PLM value
`7.59915673597e-5`. The analytic-profile fits have maximum relative L2 residuals
of 0.345 and 0.419 for PPM and xPPM, respectively; these are substantial departures
from a single spreading-ring solution. The time-dependent fitted tau values are
also less regular than for PLM. The fitted numbers should therefore be treated
as nominal effective descriptors rather than clean constant viscosities.

Initial density profiles are bitwise identical across all three reconstructions;
driver, input, and executable hashes match. Executed commands confirm orders 3
and 5. Local recomputation matches the remote viscosities within relative error
`4.6e-9` (the SciPy/platform results are not bitwise identical).
`build/2d/joseph-reconstruction-campaign/reconstruction_comparison.png` and its
PDF/CSV/JSON companions preserve the fit comparison.

To plot the measured radial profiles at 0, 5, and 10 orbits:

```sh
python3 src/problems/KeplerianDisk/plot_joseph_reconstruction_profiles.py \
  build/2d/joseph-ten-orbit-campaign/inviscid-128-ten-orbits \
  build/2d/joseph-reconstruction-campaign/inviscid-128-ppm-ten-orbits \
  build/2d/joseph-reconstruction-campaign/inviscid-128-xppm-ten-orbits \
  --output build/2d/joseph-reconstruction-campaign/ring-profiles
```

The top row shows the positive-x slice at y=0 used for the viscosity fits, and
the bottom row shows area-weighted annular averages. Initial profiles are shown
as dashed references. Intermediate times are interpolated between saved profiles,
without radial resampling; the ten-orbit CSVs match the final raw data exactly.
Per-scheme CSVs and a summary with interpolation brackets and input hashes are
saved alongside the PNG and PDF.
