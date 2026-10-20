# Particles, star formation and feedback

> **Warning: Beta feature**
>
> The particles, star-formation, and feedback module has not yet been exercised in a published science application with Quokka and should currently be treated as **beta**.
>

All particle features can only be activated when compiled with `-DAMReX_SPACEDIM=3`.

## Sink Particle Type

Sink particles carry the following real-valued attributes:

- **Mass**: Current particle mass (\\(M\\))
- **Velocity**: Three-dimensional velocity vector \\((v_x, v_y, v_z)\\)

Unlike StochasticStellarPop particles, sink particles have no evolution stage or birth/death metadata. They represent unresolved gravitationally-collapsed objects that accrete mass from the surrounding gas.

## Sink Particle Formation

### Overview

The sink formation module creates sink particles from gas cells that violate the Jeans criterion (Truelove et al. 1997). The algorithm is operator-split from the hydrodynamics and runs once per hydro timestep at the finest AMR level.

### Formation criteria

A sink particle is created in a cell if **all** of the following conditions are met:

1. **Jeans instability**: The cell density exceeds the local Jeans density,

<script type="math/tex; mode=display">
\rho > \rho_J = J^2 \frac{\pi c_{\text{eff}}^2}{G \Delta x^2},
</script>

where \\(J = 0.25\\) is the Jeans number and \\(c_{\text{eff}} = c_s \sqrt{1 + 0.74 / \beta}\\) is the effective sound speed accounting for magnetic pressure support (\\(\beta = P_{\text{thermal}} / P_{\text{magnetic}}\\)). For non-MHD simulations, \\(\beta \to \infty\\) and \\(c_{\text{eff}} = c_s\\).

2. **Not in an existing accretion zone**: The cell is not already being accreted by an existing sink particle (i.e., the accretion rate in the cell is zero).

3. **Local density maximum**: The cell has the highest density among all cells within a sphere of radius \\(3 \Delta x\\). This prevents multiple sink particles from forming in close proximity.

### Mass and momentum assignment

When a sink particle forms, it receives the excess mass above the Jeans density:

<script type="math/tex; mode=display">
m_{\text{particle}} = (\rho - \rho_J) \, \Delta x^3
</script>

The particle inherits the gas velocity of the parent cell. The cell density is then set to \\(\rho_J\\), and all conserved quantities (momentum, energy, internal energy) are scaled by the factor \\(\rho_J / \rho\\).

### Practical considerations

- The local maximum check uses a non-strict inequality (\\(\rho_{\text{cell}} \ge \rho_{\text{neighbor}}\\)), so ties do not prevent formation. In practice, exact density ties only occur in artificial initial conditions.
- Sink formation is applied after sink accretion in each timestep, so newly formed particles do not overlap with existing accretion zones.

## Sink Particle Accretion

### Overview

Sink particle accretion follows the Bondi-Hoyle-Lyttleton prescription of Krumholz et al. (2004). Gas is removed from cells surrounding each sink particle and added to the particle's mass, with the accretion rate set by the local gas properties and the particle mass.

### Accretion rate

The Bondi-Hoyle radius is

<script type="math/tex; mode=display">
r_{\text{BH}} = \frac{G M}{v_\infty^2 + c_{f,\infty}^2},
</script>

where \\(M\\) is the particle mass, \\(v_\infty\\) is the gas velocity relative to the particle (mass-weighted mean over the accretion zone), and \\(c_{f,\infty}\\) is the fast magnetosonic speed. We take the upper bound of the fast speed (when the wave propagates perpendicular to the magnetic field): \\(c_f^2 = c_s^2 + v_A^2 = c_s^2 (1 + 2/\beta)\\). For non-MHD, this reduces to \\(c_f = c_s\\).

The accretion rate is

<script type="math/tex; mode=display">
\dot{M} = 4 \pi \rho_\infty r_{\text{BH}}^2 \sqrt{v_\infty^2 + \lambda^2 c_{f,\infty}^2},
</script>

where \\(\rho_\infty\\) is the mean gas density in the accretion zone and \\(\lambda = e^{3/2}/4\\).

### Accretion kernel

The accretion zone is a sphere of radius \\(r_{\text{acc}} = 3 \Delta x\\) centered on the particle. Within this zone, each cell receives a Gaussian weight:

<script type="math/tex; mode=display">
w = \exp\left(-r^2 / r_K^2\right),
</script>

where \\(r\\) is the distance from the particle to the cell centre. The kernel radius \\(r_K\\) is resolution-adaptive:

<script type="math/tex; mode=display">
r_K = \begin{cases}
\Delta x / 4 & r_{\text{BH}} < \Delta x / 4, \\
r_{\text{BH}} & \Delta x / 4 \leq r_{\text{BH}} \leq r_{\text{acc}} / 2, \\
r_{\text{acc}} / 2 & r_{\text{BH}} > r_{\text{acc}} / 2.
\end{cases}
</script>

The accretion rate deposited in each cell is \\(-\dot{M} \, w_i / \sum_i w_i\\) (negative means accreting). Optionally, a uniform kernel (\\(w = 1\\) for all cells in the \\((7 \Delta x)^3\\) stencil) can be used for testing by setting `particles.sink_particle_use_uniform_kernel = 1`.

### Accretion limiters

The following corrections are applied to the per-cell accretion rate, in order:

1. **Mass removal cap**: No more than 25% of a cell's mass may be removed in a single timestep (Krumholz et al. 2004). This prevents artificial sound waves from being launched by rapid density changes.

2. **Jeans density floor**: If the post-accretion cell density would still exceed the Jeans density \\(\rho_J\\), the accretion rate is increased so that the final density equals \\(\rho_J\\). This is safe because such cells are at the centre of highly supersonic convergence and are causally disconnected from their surroundings.

3. **Configured density and Alfvén-speed floors**: Accretion cannot reduce a cell below its configured local density floor. For MHD runs with a positive `particles.sink_max_alfven_speed`, the final density is also constrained by \\(\rho \geq B'^2/v_{A,\max}^2 = 2E_B/v_{A,\max}^2\\), where Quokka stores \\(B'=B/\sqrt{4\pi}\\). A negative value disables the Alfvén limiter.

### Momentum accretion

When gas is accreted, the particle gains both mass and momentum. The new particle velocity is computed from momentum conservation:

<script type="math/tex; mode=display">
\vec{v}_{\text{new}} = \frac{m_{\text{old}} \vec{v}_{\text{old}} + \Delta m \, \vec{v}_{\text{gas}}}{m_{\text{old}} + \Delta m},
</script>

where \\(\Delta m\\) is the total accreted mass and \\(\vec{v}_{\text{gas}}\\) is the mass-weighted gas velocity over the accreted cells. The gas state is then updated by scaling all conserved quantities by \\((1 + \dot{M}_{\text{cell}} \Delta t / (\rho V))\\), which preserves the gas velocity field.

### Galilean invariance

The accretion rate is computed using gas velocities in the particle frame (\\(\vec{v}_\infty = \vec{v}_{\text{gas}} - \vec{v}_{\text{particle}}\\)), ensuring Galilean invariance.

### Runtime Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles.sink_particle_use_uniform_kernel` | Boolean | `0` | Use uniform accretion kernel (for testing) |
| `particles.sink_max_alfven_speed` | Float | `-1.0` | Maximum post-accretion Alfvén speed in cm/s. A negative value disables the limiter; zero is invalid. |

### Examples

#### ParticleSinkFormation Test

The `ParticleSinkFormation` test validates combined sink particle formation and accretion. The test checks:

1. Exactly one star forms in the first timestep.
2. Mass is conserved to machine precision during sink formation.
3. Mass is conserved to machine precision during sink accretion.

#### ParticleSink Test

The `ParticleSink` test validates Bondi-Hoyle accretion, Galilean invariance, density floors, and the Alfvén-speed limiter. It runs in five phases:

1. **Base simulation**: Runs with zero boost velocity and validates the density profile against an analytical solution.
2. **Boosted simulation**: Runs with a boost velocity of \\(10^8\\) cm/s and verifies that the density profile matches the analytical solution, demonstrating Galilean invariance.
3. **Multi-timestep evolution**: Continues the boosted simulation for additional timesteps and validates total mass conservation to machine precision.
4. **Parser-derived density floor**: Applies sink accretion and verifies that the minimum density equals the spatially configured floor.
5. **Alfvén-speed limiter**: Enables a limiting speed whose implied density floor exceeds the parser floor, then validates both the final minimum density and total mass conservation.

## StochasticStellarPop Particle Type

StochasticStellarPop particles carry the following real-valued attributes:

- **Mass**: Current particle mass (\\(M_{\star}\\))
- **Velocity**: Three-dimensional velocity vector \\((v_x, v_y, v_z)\\)
- **Birth time**: Simulation time when the particle formed
- **Death time**: When the particle explodes as a supernova
- **Mass at birth**: Initial mass before mass loss
- **Luminosity**: Radiation luminosity (if radiation is enabled)

Each particle also stores an integer **evolution stage** that tracks its lifecycle:

- `SNProgenitor`: High-mass star (\\(M > 8 M_{\odot}\\)) that will explode as a supernova
- `SNRemnant`: Compact remnant left after supernova explosion
- `LowMassStar`: Low-mass star that will not explode (not used in the current star formation implementation)
- `LowMassComposite`: Composite particle representing a population of low-mass stars

### Stellar luminosity table

When radiation is enabled, the luminosity of a StochasticStellarPop particle is read from a table instead of being computed from a closed-form law. The table is a [DataTable](data_table.md) with two inputs and one output per radiation group: the first input is the stellar age in years, the second is the stellar mass in solar masses, and each output is a band luminosity in erg/s. Quokka checks these names and units when it loads the file, so they must be written exactly as `age`, `mass`, `year`, `Msun`, and `erg/s`. Set the file with `particles.rad_table` and the interpolation of the outputs with `particles.rad_table_output_transform`; see [Runtime parameters](parameters.md).

The script `scripts/python/slug_luminosity_table_for_quokka.py` builds such a table from the [slug2](https://bitbucket.org/krumholz/slug2) stellar population synthesis code. It runs slug2's `write_isochrone` utility with one top-hat filter per radiation group, then converts the output to the CSV format read by `CSVReader`. Point the script at a slug2 installation with `--slug-path`, or with the `slug2_path` environment variable:

```bash
export slug2_path="/path/to/slug2"

# two bands, given as photon-energy ranges in eV
scripts/python/slug_luminosity_table_for_quokka.py PE-and-LW.csv --eV 6 11.2 --eV 11.2 13.6

# one band, given as a wavelength range in Angstroms
scripts/python/slug_luminosity_table_for_quokka.py single-band.csv --lambda 1000 1200

# a coarse FUV + Lyman continuum table on an explicit grid
scripts/python/slug_luminosity_table_for_quokka.py FUV-and-LyC.csv --eV 6 13.6 --eV 13.6 54.4 \
    --m0 2.1 --m1 120 --nm 21 --t0 1e5 --t1 1e8 --nt 31
```

Bands appear in the table in the order the `--eV` and `--lambda` options are given, so the first option becomes radiation group 0. They must be given in order of increasing photon energy, and adjacent bands must share a boundary: that is the layout of Quokka's radiation-group energy edges. A gap such as `--eV 6 11 --eV 11.2 13.6`, or a sequence that goes to lower energy, is rejected. The defaults cover ages from \\(10^5\\) to \\(2 \times 10^8\\) yr and masses from 2.1 to 120 \\(M\_\odot\\) on the `mist_2016_vvcrit_40` track set. Earlier ages are not useful: the tracks do not extend there, and the properties of a star that young depend on its accretion history rather than on its mass alone.

`write_isochrone` samples both axes logarithmically, which is why the generated table declares `log` spacing for both inputs. Two features of its output are handled by the script. A star that no longer exists at a given age is printed as `--`, and is written to the table as zero luminosity. A band with essentially no flux can integrate to a small negative number, which is zero to within the tolerance of the numerical integration, and is also floored. Because a table produced this way contains zeros, use `particles.rad_table_output_transform = "linear"`. For a log transform, pass `--floor` with a small positive value instead.

The `ParticleRadiationSlug` test validates this path end to end. A single 120 \\(M\_\odot\\) star of age 1 Myr, placed exactly on a node of both table axes, radiates into an FUV band (6 to 13.6 eV) and a Lyman continuum band (13.6 to 54.4 eV); the test checks that the radiation energy deposited on the grid equals the luminosity that slug2 prints for that star multiplied by the timestep.

## Star Formation

### Overview

The star formation module adds star particles through a stochastic prescription that plugs into the Stochastic Stellar Population specialisation.

- Runs once per hydro timestep and evaluates each cell independently.
- Always spawns a `LowMassComposite` particle when star formation is triggered.
- Adds high-mass particles (\\(> 8~M_{\odot}\\)) probabilistically so the Chabrier (2005) initial mass function (IMF) is satisfied in expectation.

### Jeans instability filter

Eligible cells are first identified through a Jeans-length check before any stochastic sampling occurs.

- Compute the Jeans density \\(\rho_J = J^2 \pi c_{\text{eff}}^2 / (G \Delta x^2)\\), where \\(J = 0.25\\) is the Jeans number and \\(c_{\text{eff}} = c_s \sqrt{1 + 0.74 / \beta}\\) is the effective sound speed, accounting for thermal pressure and magnetic pressure with \\(\beta = P_{\text{thermal}} / P_{\text{magnetic}}\\) (Mouschovias & Spitzer, 1976, ApJ, 210, 326; see also Myers et al, 2013, ApJ, 766, 97).
- Mark the cell as eligible when \\(\rho > \rho_J\\).
- Only eligible cells continue to the sampling steps below.

### Controlling the formation rate

Two efficiency parameters tune how aggressively eligible gas is converted into star particles during each hydro step.

- \\(\epsilon_{ff}\\): efficiency per free-fall time, defined through \\(\epsilon_{ff} = (\dot M_{\star} \, t_{ff}) / (M_{cell} \, \Delta t)\\).
- \\(\epsilon_{\star}\\): fraction of the cell mass used when a particle is spawned.
- The target stellar mass for the step is \\(\epsilon_{ff} M_{cell} (\Delta t / t_{ff})\\).
- Bernoulli probability for spawning: \\( P = \frac{\epsilon_{ff}}{\epsilon_{\star}} \frac{\Delta t}{t_{ff}} \\).
- The expectation value \\(\langle M_{\star} \rangle = P \epsilon_{\star} M_{cell}\\) matches the target mass provided \\(\Delta t < t_{ff}\\); the CFL condition typically enforces that inequality.

### Sampling the stellar population

Once a cell passes the filter and the Bernoulli draw succeeds, we construct the composite stellar population represented by the spawned particles.

- Every accepted draw creates one low-mass particle that represents all stars with \\(M < 8 M_{\odot}\\).
- High-mass stars follow the Chabrier (2005) IMF: log-normal below \\(1 M_{\odot}\\) and a slope of \\(2.35\\) above it.
- Pre-computed IMF integrals provide the mass fraction \\(f_{\star,high}\\) and mean mass \\(\langle m \rangle_{\star,high}\\) of the high-mass component.
- The expected number of massive stars is \\(f_{\star,high} \epsilon_{\star} M_{cell} / \langle m \rangle_{\star,high}\\); a Poisson variate with that mean sets the actual count.
- Each massive star draws its mass from the high-mass end of the IMF, while the low-mass particle retains the remaining fraction \\(1 - f_{\star,high}\\) of the spawned mass.
- If the Poisson draw returns zero, only the low-mass particle is inserted and it inherits the local gas velocity.

### Assigning particle velocities

Sampled star particles receive velocities that combine the local bulk flow with an isotropic runaway kick.

- Each massive star draws a speed from a power-law distribution \\(p(v) \propto v^{-1.8}\\) truncated between 3 and 385 km s\\(^{-1}\\), then converts it to cgs units.
- A random direction is chosen by sampling \\(\cos \theta\\) uniformly in \\([-1, 1]\\) and \\(\phi\\) in \\([0, 2\pi)\\); the resulting kick is added to the gas velocity of the parent cell.
- The total momentum of all massive stars is accumulated, and the low-mass composite particle receives the opposite momentum so that the cell-level particle system conserves momentum.

### Practical considerations

A few implementation notes help interpret corner cases and limitations of the current recipe.

- Star formation is operator-split from the hydrodynamics. When \\(t_{ff}\\) is unresolved (\\(\Delta t \gtrsim t_{ff}\\)), the true star formation rate is not captured, and this scheme provides one possible approximation; no explicit limiter is enforced beyond the CFL-controlled hydro step.
- All spawned particles are inserted at the cell centre. Other physics modules are responsible for any subsequent repositioning or feedback coupling.

## Empirically motivated early feedback

The optional empirically motivated early-feedback (EMEF) model follows Keller, Kruijssen & Chevance (2022). For a `StochasticStellarPop` particle with birth mass \\(M_{\\rm birth}\\), birth time \\(t_{\\rm birth}\\), step start \\(t\\), and timestep \\(\\Delta t\\), it requests the finite momentum increment

<script type="math/tex; mode=display">
\Delta p = \alpha p_0 M_{\rm birth}\left[x_1^{4\alpha-1}-x_0^{4\alpha-1}\right],
\qquad
x_0 = {\rm clamp}\!\left(\frac{t-t_{\rm birth}}{t_{\rm FB}},0,1\right),
\qquad
x_1 = {\rm clamp}\!\left(\frac{t+\Delta t-t_{\rm birth}}{t_{\rm FB}},0,1\right).
</script>

This is already a timestep-integrated impulse and is not multiplied by \\(\\Delta t\\) again. At \\(t=t_{\\rm birth}+t_{\\rm FB}\\), the integrated momentum is \\(\\alpha p_0 M_{\\rm birth}\\). The default values are \\(p_0=377\\) km s\\(^{-1}\\), \\(t_{\\rm FB}=3.3\\) Myr, and \\(\\alpha=1\\).

All valid stellar-population representations participate while their age interval overlaps \\([0,t_{\\rm FB}]\\): `LowMassComposite`, `SNProgenitor`, `HighMassNonExploding`, and `SNRemnant`. This stage-independent rule makes a formation event's EMEF budget proportional to the sum of its represented birth masses; later SN mass loss does not reduce the budget. Particle splitting divides both current mass and `mass_at_birth` among the children.

EMEF uses the same normalized three-cell spherical weights and four-ghost-cell requirement as SN feedback, but it constructs corrected radial vectors for each particle. Before applying the cell-by-cell velocity limiter described below, subtracting the weighted mean direction and renormalizing the vector magnitudes enforces, to deposition roundoff,

<script type="math/tex; mode=display">
\sum_j \Delta \boldsymbol{p}_j = 0,
\qquad
\sum_j \left|\Delta \boldsymbol{p}_j\right| = \Delta p.
</script>

It injects no mass, metals, passive scalars, or radiation energy. After all particle events have been accumulated, each cell's total energy receives the exact kinetic-energy change associated with its net momentum change, leaving internal energy unchanged. For each individual event, negative work \\(\\sum_j \\boldsymbol{v}_j\\!\\cdot\\!\\Delta\\boldsymbol{p}_j\\) is added as thermal energy to the particle's host cell. Cancellation between separate particle events is represented by the aggregated kinetic-energy update; it does not receive an additional inter-event thermalization term.

After simultaneous particle contributions have been accumulated, each affected cell independently caps its post-deposition bulk velocity at the larger of `particles.EMEF_max_velocity_kmps` and its pre-feedback speed. The default limit is 3000 km s\\(^{-1}\\). Gas already above the configured limit is therefore not reduced to that limit by EMEF. This post-hoc operation rescales the cell's total momentum and removes the corresponding kinetic energy while leaving internal energy unchanged. Because different cells generally receive different scale factors, this limiter deliberately gives up the uncapped stencil's exact zero-net-vector-momentum and uniform-boost invariants in exchange for preventing isolated low-density cells from controlling the hydrodynamic timestep. The limit is defined in the simulation frame.

### Deliberate differences from the published deposition algorithm

Quokka adopts equation 10 and the observational parameters of Keller, Kruijssen & Chevance (2022), but its mesh coupling is an accepted numerical adaptation rather than an exact reproduction of the paper's AREPO implementation:

- The paper distributes an event over the host cell's face-sharing neighbours, weighted by shared face area and directed along the face normals. Quokka instead reuses its normalized three-cell spherical feedback kernel. For an off-centre particle this kernel can deposit into the host cell and into cells that do not share a face with it, so the coupling radius and angular distribution differ from the published algorithm.
- The paper's face-area vectors have zero sum by the divergence theorem. Before velocity limiting, Quokka obtains the same zero-net-vector and prescribed-scalar-momentum invariants by subtracting the kernel's weighted mean direction and renormalizing the corrected vectors. The corrected vectors therefore need not coincide with Cartesian face normals or exact cell-centre radial directions.
- Quokka evaluates \\(\\sum_j \\boldsymbol{v}_j\\!\\cdot\\!\\Delta\\boldsymbol{p}_j\\) once per particle and thermalizes it in the host cell only when this event-wide sum is negative. Negative work in one recipient cell can therefore be offset by positive work in another. Momentum increments from different particles are buffered together, and cancellation between those increments is not separately thermalized. The paper does not specify simultaneous-event ordering in enough detail to establish a unique inter-event convention.

These choices are intentional: they retain Quokka's existing feedback support scale and GPU-friendly buffered deposition while enforcing the paper's total scalar- and vector-momentum constraints whenever the velocity limiter is inactive. They can nevertheless change small-scale coupling and heating relative to the published face-weighted scheme. The focused `ParticleEarlyFeedback` tests validate equation 10, the uncapped momentum and uniform-boost invariants, energy-state consistency, the adopted event-wide negative-work convention, and capped states with and without a pre-existing uniform velocity; they do not claim cell-by-cell equivalence with the paper's deposition geometry or cancellation treatment.

The operation occurs after end-of-step particle drift and star formation, and before SN deposition. A newborn particle therefore receives the clipped increment for the current \\([t,t+\\Delta t]\\) interval at its end-of-step position. This is a first-order operator split.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles.EMEF_enabled` | Boolean | `0` | Enable early feedback for `StochasticStellarPop` particles |
| `particles.EMEF_p0_kmps` | Real | `377.0` | Momentum normalization \\(p_0\\) in km s\\(^{-1}\\) |
| `particles.EMEF_tFB_Myr` | Real | `3.3` | Feedback duration in Myr |
| `particles.EMEF_alpha` | Real | `1.0` | Expansion exponent; must lie in \\([0.5,1.0]\\) |
| `particles.EMEF_max_velocity_kmps` | Real | `3000.0` | The post-feedback speed limit in km/s. Applied cell-by-cell as `max(v0, EMEF_max_velocity_kmps)`, where `v0` is the pre-feedback speed in km/s |

On a multilevel hierarchy, EMEF follows the existing SN-feedback AMR policy: feedback is deposited only from particles stored on the finest AMR level, and particles stored on coarser levels are ignored. Both paths use a same-level feedback buffer without explicit coarse-fine source synchronization, so deposition can be inconsistent when a stencil crosses a coarse-fine boundary. An active particle whose stencil is not representable inside its particle grid or the physical domain still causes an abort. EMEF is independent of `particles.disable_SN_feedback` and `particles.SN_scheme`, so EMEF-only and SN-only controls are both available.

## Supernova Feedback

### Overview

Quokka includes a particle framework for modeling stellar populations and their feedback effects on the interstellar medium (ISM). The primary feedback mechanism is supernova (SN) explosions, which inject energy and momentum into the surrounding gas. The SN module implements several physically-motivated schemes that adapt to the local resolution and density conditions.

### Supernova Feedback

#### Physical Parameters

When a progenitor star reaches its death time, it explodes as a Type II supernova with the following canonical parameters:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Blast energy | \\(E_{\text{blast}}\\) | \\(10^{51}\\) erg | Thermal energy injected into the ISM |
| Ejecta mass | \\(m_{\text{ej}}\\) | \\(10 M_{\odot}\\) | Mass expelled during explosion |
| Terminal momentum | \\(p_{\text{snr},0}\\) | \\(2.8 \times 10^5 M_{\odot} \, \text{km s}^{-1}\\) | Asymptotic momentum of the SNR (configurable via `particles.SN_p_term_Msunkmps`) |
| Remnant mass | \\(m_{\text{dead}}\\) | \\(\geq 1.4 M_{\odot}\\) | Mass of the compact remnant |
| Kinetic energy | \\(E_{\text{kin}}\\) | \\(0.5 m_{\text{ej}} v_{\text{star}}^2\\) | Kinetic energy of the ejecta |

The terminal momentum is density-dependent and scales as:

<script type="math/tex; mode=display">
p_{\text{snr}} = p_{\text{snr},0} \, n_{\text{H}}^{\alpha_p}
</script>

where \\(n_{\text{H}}\\) is the ambient hydrogen number density averaged over the deposition kernel and \\(\alpha_p = -0.17\\) by default (configurable via `particles.SN_p_term_exponent`).

#### Deposition Kernel

SN feedback is deposited into a \\((2 \times 3 + 1)^3 = 343\\) cell cubic stencil centered on the particle's location. The kernel weights are pre-computed to approximate a spherical distribution with radius \\(r = 3 \Delta x\\), where \\(\Delta x\\) is the cell width, around the host cell center. The kernel weights \\(W_{ijk}\\) are normalized such that \\(\sum_{i,j,k} W_{ijk} = 1\\).

#### Resolution-Adaptive Schemes

The feedback implementation uses the **resolution factor** \\(R_M\\) to determine whether the Sedov-Taylor phase is resolved:

<script type="math/tex; mode=display">
R_M = \frac{M_{\text{SNR}}}{M_{\text{sf}}}
</script>

where:
- \\(M_{\text{SNR}} = M_{\text{gas}} + m_{\text{ej}}\\) is the total SNR mass (gas in stencil plus ejecta)
- \\(M_{\text{sf}} = 1679 M_{\odot} \, n_{\text{H}}^{-0.26} \left(p_{\text{snr},0} / p_{\text{snr},0}^{\text{canonical}}\right)^2\\) is the shell-formation mass, scaled so that the kinetic energy \\(p_{\text{snr}}^2 / (2 M_{\text{sf}})\\) is invariant when \\(p_{\text{snr},0}\\) is changed

When \\(R_M < 1\\), the Sedov-Taylor phase is resolved and the blast wave dynamics can be captured. When \\(R_M \geq 1\\), the resolution is insufficient and the code transitions to momentum-dominated feedback following Kim & Ostriker (2017).

##### Available Schemes

Quokka provides four SN feedback schemes controlled by `particles.SN_scheme`:

1.  **`SN_thermal_only`**: Pure thermal energy injection

    - Deposits \\(E_{\text{blast}} + E_{\text{kin}}\\) into gas total energy
    - Deposits ejecta mass \\(m_{\text{ej}}\\) and momentum \\(m_{\text{ej}} \vec{v}_{\text{ej}}\\) into gas momentum. No further momentum injection
    - Simplest scheme, appropriate when Sedov-Taylor phase is well-resolved. Should only be used for testing.

2.  **Thermal and momentum schemes**: The following schemes include resolution-dependent momentum injection. In these schemes, an energy of \\(E_{\text{blast}}\\) + \\(E_{\text{kin}}\\) is injected into the gas total energy, and a fraction \\(f\\) of the asymptotic momentum \\(p_{\text{snr}}\\) is injected as momentum: \\(p_{\text{inject}} = f \, p_{\text{snr}}\\). The difference between the injected total energy and kinetic energy remains as thermal energy. Note that there is always some amount of thermal energy injected, even when \\(f = 1\\). The momentum fraction \\(f\\) is determined by the resolution factor \\(R_M\\) as follows:

    a. **`SN_thermal_or_thermal_momentum`** (default):

       - When \\(R_M < 0.027\\): Pure thermal (like `SN_thermal_only`) with \\(f = 0\\)
       - When \\(0.027 \leq R_M < 1\\): Partial momentum injection with \\(f = 0.529 \sqrt{R_M}\\). Based on Kim & Ostriker (2017) calibration, \\(f^2 = 0.28 R_M\\) means that 28% of the total energy is kinetic energy in this case.
       - When \\(R_M \geq 1\\): Full momentum injection with \\(f = 1\\)

    b. **`SN_thermal_kinetic_or_thermal_momentum`**:

       - When \\(R_M < 0.027\\): Pure kinetic injection with \\(f = \sqrt{2 R_M}\\)
       - When \\(0.027 \leq R_M < 1\\): Partial momentum injection with \\(f = 0.529 \sqrt{R_M}\\)
       - When \\(R_M \geq 1\\): Full momentum injection with \\(f = 1\\)

    c. **`SN_pure_kinetic_or_thermal_momentum`**:

       - Always injects full terminal momentum regardless of resolution with \\(f = 1\\)

#### Momentum Deposition

For schemes with momentum injection, the following momentum deposition procedure guarantees Galilean invariance for a single SN event:

1. **Compute COM velocity** of the SNR (gas + ejecta):

<script type="math/tex; mode=display">
\vec{v}_{\text{COM}} = \frac{\vec{P}_{\text{gas}} + m_{\text{ej}} \vec{v}_{\text{ej}}}{M_{\text{gas}} + m_{\text{ej}}}
</script>

where \\(\vec{P}_{\text{gas}}\\) is the total momentum of gas in the stencil (kernel-weighted).

2. **Deposit momentum** such that each cell receives:

<script type="math/tex; mode=display">
\Delta \vec{p}_{ijk} = \left[\rho_{\text{new}} \vec{v}_{\text{COM}} - \vec{p}_{\text{old}}\right] + \vec{p}_{\text{radial}}
</script>

where \\(\vec{p}_{\text{radial}} = f \, p_{\text{snr}} \, W_{ijk} \, \hat{\mathbf{r}}_{ijk}\\) is the radial momentum component.

3. **Energy deposition** includes a cross term:

<script type="math/tex; mode=display">
\Delta E_{ijk} = \left(E_{\text{blast}} + E_{\text{kin}}\right) W_{ijk} + \vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial}}
</script>

The cross term <script type="math/tex">\vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial}}</script> accounts for the kinetic energy change from the radial expansion, ensuring Galilean invariance. This term sums to zero over all cells in the stencil, (momentum is conserved), ensuring energy conservation.

**Background smoothing**. To ensure the cross term cancels when summing over all cells in the stencil, the velocity field (but not the density) must be smoothed such that <script type="math/tex">\sum_i \vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial},i} = \vec{v}_{\text{COM}} \cdot \sum_i \vec{p}_{\text{radial},i} = 0</script>. In our tests, this smoothing also dramatically reduces peak velocities in SNR. In a shearing/turbulent background, this artificially homogenizes the gas velocity and changes kinetic energy unrelated to the SN. We provide a parameter `particles.SN_smooth_gas_velocity=0` to turn off this smoothing: <script type="math/tex">\Delta \vec{p}_{ijk} = m_{\text{ej}} \vec{v}_{\text{ej}} / V + \vec{p}_{\text{radial}}</script> and <script type="math/tex">\Delta E_{ijk} = \left(E_{\text{blast}} + E_{\text{kin}}\right) W_{ijk} + \vec{v}_{i} \cdot \vec{p}_{\text{radial}}</script>. In this case, the cross term sums to a non-zero value: <script type="math/tex">\sum_i \vec{v}_i \cdot \vec{p}_{\text{radial},i} = (\vec{v}_{\text{COM}} + \delta v_i) \cdot \sum_i \vec{p}_{\text{radial},i} = \sum_i \delta v_i \cdot \vec{p}_{\text{radial},i}</script>, where <script type="math/tex">\delta v_i</script> is the velocity of cell <script type="math/tex">i</script> relative to the COM frame.

### Runtime Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles.SN_scheme` | String | `SN_thermal_or_thermal_momentum` | Feedback scheme (see above) |
| `particles.SN_p_term_Msunkmps` | Float | `2.8e5` | Terminal momentum \\(p_{\text{snr},0}\\) in units of \\(M_\odot\,\mathrm{km\,s}^{-1}\\). The shell-formation mass \\(M_\mathrm{sf}\\) is scaled as \\((p/p_\mathrm{canonical})^2\\) to preserve the kinetic energy \\(p^2/(2M_\mathrm{sf})\\). |
| `particles.SN_p_term_exponent` | Float | `-0.17` | Exponent \\(\alpha_p\\) of the ambient-density scaling of the terminal momentum, \\(p_{\text{snr}} = p_{\text{snr},0} \, n_{\text{H}}^{\alpha_p}\\). |
| `particles.disable_SN_feedback` | Boolean | `0` | Disable SN feedback entirely |
| `particles.verbose` | Integer | `0` | Verbosity level for particle diagnostics |
| `particles.stellar_velocity_limit` | Float | \\(10^8\\) cm/s | Maximum allowed stellar velocity (aborts if exceeded) |
| `particles.SN_smooth_gas_velocity` | Integer | `1` | Smooth gas velocity in the stencil to enforce energy conservation |

### 	Implementation Notes

#### Operator Splitting

SN feedback is operator-split from the hydrodynamics and applied after each timestep:

1. Hydrodynamics advances the gas state by \\(\Delta t\\)
2. Particle positions are updated by drift
3. SN candidates (particles reaching death time) are identified
4. Feedback is deposited into a temporary buffer
5. Buffer is added to the state using a reproducibility-aware algorithm
6. Particle evolution stage is updated to `SNRemnant`

#### Numerical Reproducibility

The deposition uses a roundoff-resistant algorithm to ensure bit-for-bit reproducibility across different processor counts. The algorithm:

1. Deposits feedback into a temporary buffer with a count field
2. Communicates the buffer across ranks, which introduces roundoff errors due to non-associative floating-point arithmetic
3. Removes low-order bits controlled by `particles.reproducibility_roundoff_redundancy` to remove this error

#### AMR Considerations

- Feedback is always deposited at the finest level covering the particle
- If a particle sits at a coarse-fine boundary, feedback is deposited into fine-level cells (known limitation that needs to be fixed in the future)
- The reflux operation ensures conservation across AMR boundaries
- The kernel stencil size is fixed in cell widths, so the physical size adapts to local resolution
- The use of `setForceFinestLevel(true)` is recommended to ensure that all SN progenitors, and thus SN events, exist at finest level. 

#### Limitations

- Accurate radiative cooling must be present to resolve the pressure-confined snowplow phase
- The spherical kernel is approximated on a Cartesian grid, introducing mild anisotropy
- The terminal momentum formula assumes solar metallicity and Type II SNe physics
- No explicit treatment of metal enrichment (can be added via passive scalars)

### Physical Motivation

The resolution-dependent momentum injection is based on the realization that under-resolved SN explosions lose energy to artificial radiative losses within a few timesteps. The momentum-based schemes compensate by directly injecting momentum when the Sedov-Taylor phase cannot be captured.

The terminal momentum \\(p_{\text{snr}}\\) represents the asymptotic momentum that a supernova remnant reaches in the pressure-confined snowplow phase, after most of the kinetic energy has been radiated away. This value (Kim & Ostriker 2017) is calibrated from high-resolution simulations and depends weakly on the ambient density.

The \\(R_M\\) factor effectively measures whether the simulation has sufficient resolution and density to capture the Sedov-Taylor expansion. When \\(R_M \ll 1\\), the blast wave will expand through many cells before reaching the shell-formation radius, allowing the hydrodynamics to naturally capture the momentum buildup. When \\(R_M \sim 1\\) or larger, the explosion is under-resolved and the code preemptively injects the expected terminal momentum.

### Examples

#### Basic SN Test

The `SN` problem provides a canonical test with one SN progenitor in a uniform medium:

```
particles.SN_scheme = SN_thermal_or_thermal_momentum
problem.boost_vel_x = 1.0e8 # boost velocity in x direction
```

This test is run three times, with ambient density of \\(10, 1.0\\), and \\(0.1 ~\mathrm{cm}^{-3}\\), corresponding to a shell-formation radius of 7, 22, and 70 pc, respectively. The spatial resolution is 5 pc. The default `SN_thermal_or_thermal_momentum` scheme is used. The three ambient densities implies \\(R_M \approx (3 \Delta x / R_{\text{sf}})^3 = 10, 0.3\\), and \\(0.01\\), respectively, placing the scheme in the under-resolved, partially resolved, and well-resolved regimes, respectively.

In each of the three tests, the simulation is run twice, one initially at rest and one with an initial boost velocity of \\(200\\) km/s, to demonstrate Galilean invariance. The temperature and velocity profiles are identical in the boosted and rest frames (to some tolerance).

#### Random Blast Test

The `RandomBlast` problem provides a testbed for multiple SN explosions with various ambient conditions and bulk flows.

## Stochastic stellar population chemistry components

When `Particle_Traits<problem_t>::enable_chemical_feedback` is enabled, chemistry
components follow the ordinary particle fields and any luminosity components.
Their block order is `total`, `SNII`, `WR`, then `AGB`, independent of which
feedback channels are enabled at runtime.

Within each block, slots follow the configured tracked-isotope order. Names use
`chem_birth_<block>_<isotope>`, for example `chem_birth_total_C12`,
`chem_birth_SNII_C12`, `chem_birth_WR_C12`, and `chem_birth_AGB_C12`.
These fields store birth abundances, not the subsequently accumulated ejecta.

`StochasticStellarPopParticleChemistryBlockCapacity<problem_t>()` is the storage
stride of each block, equal to `numPassiveScalars` when chemical feedback storage
is enabled, and zero otherwise. It is not the number of tracked isotopes. Reserved
slots beyond the isotope list use `unused_<index>`, with a zero-based slot index;
for three tracked isotopes, the next total-block component is
`chem_birth_total_unused_3`.

## Star Particle Type (modular stellar evolution)

Star particles (`ParticleSwitch::Star`) represent individual stars whose radius and luminosity evolve through a **pluggable stellar-evolution model** selected at compile time.

### Particle attributes

Real components: `mass`, `vx`, `vy`, `vz`, `birth_time`, `death_time` (\\(-1\\) while unset), `mdot` (accretion rate, set by the accretion module), `radius` (set by the stellar model), `lum_0`, ..., `lum_<nGroups-1>` (luminosity per radiation group), followed by the model's extra real components. A model declares its extra components with `nExtraReal` / `nExtraInt` and names them with `extraRealNames` / `extraIntNames`; the integer components of a Star particle are all model-defined.

### Choosing a model

The model is selected via `Particle_Traits<problem_t>::stellar_model`, which defaults to `ToyStellarModel` through `DefaultParticleTraits`. A problem can override it by adding `using stellar_model = MyModel;` to its `Particle_Traits` specialization. A model is a struct of GPU device functions; the dispatcher `StellarUpdate::updateStellarProperties` passes the particle's real and integer data to `Model::evolve`, which stores `radius` and `lum` once per coarse step. The update runs before the accretion step of the same coarse step, so it uses the mass and accretion rate from the previous step's accretion.

### Formation

Star particles form from Jeans-unstable cells with the same criterion and mass removal as Sink particles (see [Sink Particle Formation](#sink-particle-formation)). A new particle has `birth_time` set to the current time and `death_time = -1`; all other components except mass and velocity start at zero, which the models treat as "not yet started".

### Toy model

`ToyStellarModel` is stateless:

- Radius: \\(R = R\_\odot (M/M\_\odot)^{0.4}\\)
- Stellar luminosity: \\(L\_\star = L\_\odot (M/M\_\odot)^{3.5}\\)
- Accretion luminosity: \\(L\_\mathrm{acc} = G M \dot{M} / R\\)
- Stored luminosity: \\(L = L\_\star + L\_\mathrm{acc}\\)

### Deuterium-burning model

`DeuteriumBurningStellarModel` (`src/particles/deuterium_stellar_model.hpp`) is the one-zone protostellar model of Offner et al. (2009, ApJ 703, 131, Appendix B), which builds on Nakano et al. (1995, 2000) and Tan & McKee (2004). The update reproduces the ORION2 star particle module (`StarParticleData`). The star is a polytrope of index \\(n\\) whose radius \\(r\\) follows from energy conservation, and it passes through six phases. This section contains everything needed to reimplement the model.

#### State, phases and parameters

Each particle stores its mass \\(m\\) and accretion rate \\(\dot m\\) (both set by the accretion module), the radius \\(r\\), and the model extras `mdeut` (\\(m\_d\\), the mass of accreted gas whose deuterium is not yet burned), `n` and `burn_state`. A particle that starts with all of these at zero is in `PreCollapse`.

| `burn_state` | Phase | \\(L\_D\\) |
|---|---|---|
| 0 | `PreCollapse`: model not started; \\(L = 0\\) | 0 |
| 1 | `NoBurning`: contraction before deuterium ignition | 0 |
| 2 | `CoreDeuteriumFixedTc`: core deuterium burning holds \\(T\_c\\) fixed | Eq. (9) |
| 3 | `CoreDeuteriumVariableTc`: core deuterium burned as fast as it is accreted | \\(\dot m\\,\psi\_D\\) |
| 4 | `ShellDeuterium`: radiative core, deuterium burned in a shell | \\(\dot m\\,\psi\_D\\) |
| 5 | `MainSequence`: \\(r = R\_{\rm ZAMS}(m)\\) | 0 |

| Parameter | Value |
|---|---|
| minimum mass \\(M\_{\rm min}\\) | \\(0.01\\,M\_\odot\\) |
| mean molecular weight \\(\mu\\), hydrogen mass \\(m\_H\\) | 0.613, \\(m\_p\\) |
| fraction of infall energy radiated in the inner disk \\(f\_k\\) | 0.5 |
| fraction of accretion power radiated \\(f\_{\rm acc}\\) | 0.5 |
| shell-burning threshold \\(f\_{\rm rad}\\) | 0.18 |
| radius factor entering shell burning | 2.1 |
| Hayashi temperature \\(T\_H\\), deuterium ignition temperature \\(T\_D\\) | 3000 K, \\(1.5\times10^6\\) K |
| \\(\psi\_I\\) (dissociation and ionization), \\(\psi\_D\\) (deuterium burning) | 17 eV and 100 eV per \\(m\_u\\), i.e. \\(17\\,{\rm eV}\\,N\_A\\) and \\(100\\,{\rm eV}\\,N\_A\\) per gram |
| \\(L\_\odot\\), year | \\(3.828\times10^{33}\\) erg s\\(^{-1}\\), \\(3.15576\times10^7\\) s |
| finite-difference step \\(\delta\\) | \\(10^{-3}\\) |
| sub-step limit \\(\epsilon\\), maximum number of sub-steps | 0.01, \\(10^4\\) |

Physical constants (CGS, Quokka's values): \\(G = 6.67428\times10^{-8}\\), \\(k\_B = 1.380649\times10^{-16}\\), \\(m\_p = 1.67262192595\times10^{-24}\\), \\(\sigma\_{\rm SB} = 5.670374419\times10^{-5}\\), \\(a = 4\sigma\_{\rm SB}/c\\) with \\(c = 2.99792458\times10^{10}\\), \\(M\_\odot = 1.9884\times10^{33}\\), \\(R\_\odot = 6.957\times10^{10}\\), \\(1\\,{\rm eV} = 1.602176634\times10^{-12}\\) erg, \\(N\_A = 6.02214076\times10^{23}\\).

#### Initial radius and polytropic index

With \\(\dot m\\) in \\(M\_\odot\\,{\rm yr}^{-1}\\) in both equations,

<script type="math/tex; mode=display">
r_{\rm init} = 2.5\,R_\odot\left(10^5\,\dot m\right)^{0.2}, \tag{1}
</script>

<script type="math/tex; mode=display">
n_{\rm init} = 5 - \frac{3}{1.475 + 0.07\log_{10}\left(8\times10^{-5}\,\dot m\right)}, \quad\text{clamped to } [1.5, 3]; \qquad n_{\rm init} = 1.5 \text{ if } \dot m \le 0. \tag{2}
</script>

#### Polytrope structure

<script type="math/tex; mode=display">
a_g = \frac{3}{5-n}, \qquad \rho_c = \frac{3m}{4\pi r^3 f_\rho(n)}, \qquad P_c = f_P(n)\,\frac{G m^2}{r^4}, \tag{3}
</script>

where \\(f\_\rho = \bar\rho/\rho\_c\\) and \\(f\_P\\) are interpolated linearly in \\(n\\) from Table A, on the interval \\(i = \lfloor (n - 1.5)/0.1 \rfloor\\) clamped to \\([0, 15]\\). The central temperature \\(T\_c\\) is the root of

<script type="math/tex; mode=display">
P_c = \frac{\rho_c k_B T_c}{\mu m_H} + \frac{a}{3}T_c^4 \tag{4}
</script>

on \\([0, \max(T\_{\rm gas}, T\_{\rm rad})]\\), with \\(T\_{\rm gas} = P\_c\mu m\_H/(k\_B\rho\_c)\\) and \\(T\_{\rm rad} = (3P\_c/a)^{1/4}\\); Quokka solves this and Eq. (5) with TOMS 748 to a relative tolerance of \\(10^{-12}\\), which keeps Eq. (6) accurate. The central gas-pressure fraction is \\(\beta\_c = \rho\_c k\_B T\_c/(\mu m\_H P\_c)\\).

The mean gas-pressure fraction \\(\beta(m, r, n)\\) is: for \\(n = 3\\), the root in \\([10^{-4}, 1]\\) of the Eddington quartic

<script type="math/tex; mode=display">
P_c^3 = \frac{3}{a}\left(\frac{k_B \rho_c}{\mu m_H}\right)^4 \frac{1-\beta}{\beta^4}; \tag{5}
</script>

otherwise 1 for \\(m < 5\\,M\_\odot\\), and for \\(m \geq 5\\,M\_\odot\\) bilinear interpolation, linear in \\(m\\) and \\(n\\), in Table B (below); masses beyond \\(50\\,M\_\odot\\) use the last row. Following ORION2, \\(\beta\\) jumps from 1 to 0.988 at \\(5\\,M\_\odot\\), so Eq. (6) gives \\(d\log\beta/d\log m \approx -12\\) for \\(5/(1+\delta) \le m/M\_\odot < 5\\) when \\(n < 3\\). The mass derivatives are one-sided differences at fixed \\(r\\) and \\(n\\):

<script type="math/tex; mode=display">
\frac{d\log\beta}{d\log m} = \frac{1}{\delta}\left[\frac{\beta(m(1+\delta))}{\beta(m)} - 1\right], \qquad \frac{d\log(\beta/\beta_c)}{d\log m} = \frac{1}{\delta}\left[\frac{(\beta/\beta_c)(m(1+\delta))}{(\beta/\beta_c)(m)} - 1\right], \tag{6}
</script>

and the second one is 0 for \\(n = 3\\).

#### Luminosities

The zero-age main-sequence fits of Tout et al. (1996) at solar metallicity, with \\(x = m/M\_\odot\\) and the coefficients of Table C, are

<script type="math/tex; mode=display">
\frac{L_{\rm ZAMS}}{L_\odot} = \frac{c_1 x^{5.5} + c_2 x^{11}}{c_3 + x^3 + c_4 x^5 + c_5 x^7 + c_6 x^8 + c_7 x^{9.5}}, \qquad
\frac{R_{\rm ZAMS}}{R_\odot} = \frac{d_1 x^{2.5} + d_2 x^{6.5} + d_3 x^{11} + d_4 x^{19} + d_5 x^{19.5}}{d_6 + d_7 x^2 + d_8 x^{8.5} + x^{18.5} + d_9 x^{19.5}}. \tag{7}
</script>

The accretion-shock, inner-disk, protostellar and ionization luminosities are

<script type="math/tex; mode=display">
L_{\rm acc} = f_{\rm acc} f_k \frac{G m\dot m}{r}, \quad L_{\rm disk} = (1-f_k)\frac{G m\dot m}{r}, \quad L_\star = \max\left(L_{\rm ZAMS} + L_{\rm acc},\ 4\pi r^2\sigma_{\rm SB}T_H^4\right), \quad L_I = \dot m\,\psi_I. \tag{8}
</script>

The deuterium luminosity \\(L\_D\\) depends on the phase (table above). In `CoreDeuteriumFixedTc` it is the value that keeps \\(T\_c\\) fixed (Nakano et al. 2000, Eq. 5.13):

<script type="math/tex; mode=display">
L_D = L_\star + L_I + \frac{G m\dot m}{r}\left\{1 - f_k - \frac{a_g\beta}{2}\left[1 + \frac{d\log(\beta/\beta_c)}{d\log m}\right]\right\}. \tag{9}
</script>

The stored luminosity is \\(L = L\_\star + L\_{\rm disk}\\) in `lum_0` (the other groups are zero), and \\(L = 0\\) in `PreCollapse`.

#### Radius evolution

Before the main sequence, the radius follows (Nakano et al. 2000, Eq. 5.8)

<script type="math/tex; mode=display">
\frac{dr}{dt} = \underbrace{\frac{2\dot m\,r}{m}\left(1 - \frac{1-f_k}{a_g\beta} + \frac{1}{2}\frac{d\log\beta}{d\log m}\right)}_{A} - \underbrace{\frac{2r^2}{a_g\beta\,G m^2}\left(L_\star + L_I - L_D\right)}_{B}, \tag{10}
</script>

integrated with forward Euler, \\(r \leftarrow r + h\\,(A - B)\\). For the sub-step size, the rate scale is

<script type="math/tex; mode=display">
S = \max\left(|A|,\ \frac{2r^2}{a_g\beta\,G m^2}\left(L_\star + L_I + |L_D|\right)\right). \tag{11}
</script>

Using the individual terms matters: while the star contracts, \\(A\\) and \\(B\\) nearly cancel, so \\(|A - B|\\) does not limit the step.

#### Update algorithm

Each coarse step calls `evolve` once with the step \\(\Delta t\\), the current mass \\(m\\) (already including the accreted mass) and \\(\dot m\\):

1. Set all `lum` groups to 0 and \\(t\_{\rm left} = \Delta t\\).
2. If the phase is `PreCollapse`: return if \\(m < M\_{\rm min}\\) or \\(\dot m \le 0\\). Otherwise set \\(r = r\_{\rm init}\\), \\(n = n\_{\rm init}\\) (Eqs. 1–2), the phase to `NoBurning`, and \\(t\_{\rm left} = \min(\Delta t, (m - M\_{\rm min})/\dot m)\\): the model starts when the mass passed \\(M\_{\rm min}\\).
3. Repeat while \\(t\_{\rm left} > 0\\), at most \\(10^4\\) times (any time left after that is not integrated). The mass is assumed to grow linearly in time, and each sub-step uses the mass at its end.
   1. Sub-step: \\(h = t\_{\rm left}\\). Outside the main sequence, \\(h = \min(t\_{\rm left}, \epsilon r/S)\\) with \\(S\\) (Eq. 11) evaluated with the current \\(r\\), \\(n\\) and phase at the mass at the start of the sub-step, \\(\max(m - \dot m\\,t\_{\rm left}, M\_{\rm min})\\).
   2. Sub-step mass \\(m\_s = \max(m - \dot m\\,(t\_{\rm left} - h), M\_{\rm min})\\). Outside the main sequence, \\(r \leftarrow r + h\\,(A - B)\\) with Eq. (10) at \\(m\_s\\) and the current \\(r\\), \\(n\\), phase. Then \\(t\_{\rm left} \leftarrow t\_{\rm left} - h\\).
   3. Phase update, using \\(m\_s\\) and the new \\(r\\); at most one phase change per sub-step:
      - `NoBurning`: \\(m\_d = m\_s\\), \\(n = n\_{\rm init}(\dot m)\\). If \\(T\_c > T\_D\\) (Eq. 4), go to `CoreDeuteriumFixedTc` and set \\(n = 1.5\\).
      - `CoreDeuteriumFixedTc`: \\(m\_d \leftarrow m\_d + \dot m h - L\_D h/\psi\_D\\) with \\(L\_D\\) from Eq. (9). If \\(m\_d \le \dot m h\\), the core deuterium is exhausted: go to `CoreDeuteriumVariableTc` and set \\(m\_d = 0\\).
      - `CoreDeuteriumVariableTc`: \\(m\_d = 0\\). If \\(\dot m\\,\psi\_D \le f\_{\rm rad} L\_{\rm ZAMS}\\), a radiative barrier forms: go to `ShellDeuterium`, set \\(n = 3\\) and \\(r \leftarrow 2.1\\,r\\) (once).
      - `ShellDeuterium`: \\(m\_d = 0\\). If \\(r \le R\_{\rm ZAMS}\\), go to `MainSequence` and set \\(r = R\_{\rm ZAMS}\\).
      - `MainSequence`: \\(m\_d = 0\\), \\(r = R\_{\rm ZAMS}\\).
4. Store \\(L = L\_\star + L\_{\rm disk}\\) (Eq. 8) at the final \\(m\\) and \\(r\\).

When \\(\epsilon r/S \geq \Delta t\\) there is one sub-step with \\(m\_s = m\\), and step 3 is the same as ORION2's `updateState` with \\(\Delta m = \dot m\Delta t\\). This is the case for steps small compared with the contraction time, as in ORION2. Without sub-steps, explicit steps of \\(\Delta m = 0.01\\,M\_\odot\\) at \\(10^{-5}\\,M\_\odot\\,{\rm yr}^{-1}\\) already give radii off by 20%; with them the error stays below 5% even at \\(\Delta m = 1\\,M\_\odot\\) per step. The model update runs before the accretion step of the same coarse step, so it lags the accretion by one step.

#### Tables

**Table A.** Lane-Emden factors \\(f\_\rho = \bar\rho/\rho\_c\\) and \\(f\_P = P\_c/(Gm^2/r^4)\\) (Kippenhahn & Weigert).

| \\(n\\) | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 | 2.1 | 2.2 | 2.3 |
|---|---|---|---|---|---|---|---|---|---|
| \\(f\_\rho\\) | 0.166931 | 0.14742 | 0.129933 | 0.114265 | 0.100242 | 0.0877 | 0.0764968 | 0.0665109 | 0.0576198 |
| \\(f\_P\\) | 0.770087 | 0.889001 | 1.02979 | 1.19731 | 1.39753 | 1.63818 | 1.92909 | 2.2825 | 2.71504 |

| \\(n\\) | 2.4 | 2.5 | 2.6 | 2.7 | 2.8 | 2.9 | 3.0 | 3.1 |
|---|---|---|---|---|---|---|---|---|
| \\(f\_\rho\\) | 0.0497216 | 0.0427224 | 0.0365357 | 0.0310837 | 0.0262952 | 0.0221057 | 0.0184553 | 0.01529 |
| \\(f\_P\\) | 3.24792 | 3.90921 | 4.73657 | 5.78067 | 7.11088 | 8.82286 | 11.0515 | 13.9885 |

**Table B.** Mean gas-pressure fraction \\(\beta\\) of a polytrope.

| \\(m/M\_\odot\\) | \\(n = 1.5\\) | \\(n = 2\\) | \\(n = 2.5\\) | \\(n = 3\\) |
|---|---|---|---|---|
| 5 | 0.98785 | 0.988928 | 0.98947 | 0.989634 |
| 7.5 | 0.97438 | 0.976428 | 0.977462 | 0.977774 |
| 10 | 0.957927 | 0.960895 | 0.962397 | 0.962846 |
| 12.5 | 0.939787 | 0.943497 | 0.945369 | 0.945922 |
| 15 | 0.92091 | 0.925151 | 0.927276 | 0.927896 |
| 17.5 | 0.901932 | 0.906512 | 0.908785 | 0.909436 |
| 20 | 0.883254 | 0.888017 | 0.890353 | 0.891013 |
| 22.5 | 0.865111 | 0.86994 | 0.872277 | 0.872927 |
| 25 | 0.847635 | 0.852445 | 0.854739 | 0.855367 |
| 27.5 | 0.830886 | 0.835619 | 0.837842 | 0.838441 |
| 30 | 0.814885 | 0.8195 | 0.821635 | 0.822201 |
| 32.5 | 0.799625 | 0.804095 | 0.806133 | 0.806664 |
| 35 | 0.785082 | 0.789394 | 0.791328 | 0.791825 |
| 37.5 | 0.771226 | 0.775371 | 0.777202 | 0.777665 |
| 40 | 0.758022 | 0.761997 | 0.763726 | 0.764156 |
| 42.5 | 0.745433 | 0.749238 | 0.750869 | 0.751268 |
| 45 | 0.733423 | 0.73706 | 0.738596 | 0.738966 |
| 47.5 | 0.721954 | 0.725429 | 0.726874 | 0.727216 |
| 50 | 0.710993 | 0.714311 | 0.715671 | 0.715987 |

The \\(n = 3\\) column is used only to interpolate for \\(2.5 < n < 3\\); at \\(n = 3\\) Eq. (5) is used.

**Table C.** Coefficients of Eq. (7).

| \\(c\_1\\) | \\(c\_2\\) | \\(c\_3\\) | \\(c\_4\\) | \\(c\_5\\) | \\(c\_6\\) | \\(c\_7\\) | | |
|---|---|---|---|---|---|---|---|---|
| 0.39704170 | 8.52762600 | 0.00025546 | 5.43288900 | 5.56357900 | 0.78866060 | 0.00586685 | | |
| \\(d\_1\\) | \\(d\_2\\) | \\(d\_3\\) | \\(d\_4\\) | \\(d\_5\\) | \\(d\_6\\) | \\(d\_7\\) | \\(d\_8\\) | \\(d\_9\\) |
| 1.71535900 | 6.59778800 | 10.08855000 | 1.01249500 | 0.07490166 | 0.01077422 | 3.08223400 | 17.84778000 | 0.00022582 |

#### Differences from the text of Offner et al. (2009)

The model follows the ORION2 code, which differs from the published appendix in four places: the shell-burning test is \\(L\_D \le 0.18\\,L\_{\rm ZAMS}\\) (the appendix writes \\(L\_D/L\_{\rm ms} > 0.33\\), which would start shell burning at once); \\(L\_\star\\) contains \\(L\_{\rm acc}\\) inside the Hayashi maximum, in Eqs. (9)–(10) and in \\(L\\) (the appendix uses \\(\max(L\_{\rm ms}, L\_H)\\)); Eq. (2) has the factor \\(8\times10^{-5}\\); and \\(\psi\_I\\), \\(\psi\_D\\) are 17 and 100 eV per \\(m\_u\\) instead of 2.5 and \\(15\\,L\_\odot\\) per \\(10^{-5}\\,M\_\odot\\,{\rm yr}^{-1}\\) (8% and 6% larger). In addition, \\(\beta\_c\\) uses the \\(T\_c\\) of Eq. (4), not \\(T\_D\\).

#### Plan: a model that follows Offner et al. (2009)

`DeuteriumBurningStellarModel` stays as it is, a faithful port of ORION2 that serves as the reference solution. ORION2 predates the published appendix, so a second model will follow Offner et al. (2009, Appendix B) as published. Nothing in this subsection is implemented yet.

| # | Item | ORION2 (current) | Offner et al. (2009) (planned) |
|---|---|---|---|
| 1 | initial \\(n\\), Eq. (2) | argument of the log is \\(8\times10^{-5}\\,\dot m\\) | argument is \\(\dot m\\): \\(n\_{\rm init} = 2.33\\) instead of 1.5 at \\(10^{-5}\\,M\_\odot\\,{\rm yr}^{-1}\\) |
| 2 | \\(n\\) during `NoBurning` | reset to \\(n\_{\rm init}(\dot m)\\) every sub-step | kept at its initial value ("an accreting polytrope of fixed index") |
| 3 | interior luminosity in Eqs. (9)–(10) | \\(L\_\star = \max(L\_{\rm ZAMS} + L\_{\rm acc}, L\_H)\\) | \\(L\_{\rm int} = \max(L\_{\rm ZAMS}, L\_H)\\) |
| 4 | stored luminosity | \\(L\_\star + L\_{\rm disk}\\) | \\(L\_{\rm int} + L\_{\rm acc} + L\_{\rm disk}\\) |
| 5 | \\(L\_I\\) | \\(\dot m \times 17\\,{\rm eV}/m\_u\\) | \\(2.5\\,L\_\odot\\,(\dot m/10^{-5}\\,M\_\odot\\,{\rm yr}^{-1})\\) |
| 6 | \\(L\_D\\) in phases 3–4 and in the \\(m\_d\\) update (\\(\psi\_D\\)) | \\(100\\,{\rm eV}/m\_u\\) | \\(15\\,L\_\odot\\) per \\(10^{-5}\\,M\_\odot\\,{\rm yr}^{-1}\\) |
| 7 | shell-burning test | \\(L\_D \le 0.18\\,L\_{\rm ZAMS}\\) | \\(f\_{\rm rad} = 0.33\\); the printed \\(L\_D/L\_{\rm ms} > 0.33\\) would start shell burning on the first step of phase 3 (at \\(10^{-5}\\,M\_\odot\\,{\rm yr}^{-1}\\), \\(L\_D/L\_{\rm ZAMS} \approx 460\\) there), so implement \\(L\_D \le 0.33\\,L\_{\rm ZAMS}\\) |
| 8 | core deuterium exhausted | \\(m\_d \le \dot m h\\), then \\(m\_d = 0\\) | \\(m\_d < 0\\) after the update |
| 9 | order within a step | phase change, then \\(L\\) | update \\(r\\) and \\(m\_d\\), then \\(L\\), then the phase change |

Items 3–4 matter most: at \\(0.1\\,M\_\odot\\) on the \\(10^{-5}\\,M\_\odot\\,{\rm yr}^{-1}\\) track, \\(L\_{\rm int} = 0.034\\,L\_\odot\\) while \\(L\_\star = 11.5\\,L\_\odot\\), so the contraction term \\(B\\) of Eq. (10) drops by a factor of about 340. Item 1 changes \\(a\_g\\) and the Lane-Emden factors from the start.

**Code.** Make the update algorithm a template on a parameter struct that holds the constants and switches of the table (\\(\psi\_I\\), \\(\psi\_D\\), \\(f\_{\rm rad}\\), the \\(n\_{\rm init}\\) factor, and flags for items 2–4, 8 and 9). `DeuteriumBurningStellarModel` becomes the ORION2 instance, and a new `Offner2009StellarModel` is the second instance. Both share the structure functions, tables, sub-stepping, extras and output names; a problem picks one through `Particle_Traits<problem_t>::stellar_model`.

**Validation.**

- The ORION2 instance must still pass the existing `ParticleStar` comparison with the ORION2 tracks, unchanged.
- For the Offner instance, write a short standalone Python implementation of the published equations from this page, generate constant-\\(\dot m\\) tracks at \\(10^{-5}\\) and \\(10^{-4}\\,M\_\odot\\,{\rm yr}^{-1}\\), store them in `src/problems/ParticleStar/`, and compare them in `ParticleStar` like the ORION2 tracks.
- Check the Offner tracks against the Palla & Stahler (1991, 1992) protostellar radii to which the model is calibrated (Offner et al. report about 10% agreement), and report the phase-change masses of both models side by side.

**Decisions needed.**

- Item 7: the direction of the shell-burning test (\\(\le\\) as proposed).
- Item 9: whether \\(L\\) is computed before or after the phase change.
- Whether to use the recalibrated parameters of Hosokawa & Omukai (2009), which Offner et al. recommend for future work; this would be a third parameter set.

### Validation tests

`ParticleStarEvolution` places one Star particle in a uniform medium and lets it accrete via the grid Bondi accretion module (small-mass regime, \\(\dot{M}\approx\\) const). It asserts, each step, that the particle's stored radius and luminosity match the closed-form laws of the toy model within ~2% (the tolerance absorbing the one-timestep lag between accretion and the stellar update), and reports the mean accretion rate against the analytic Bondi value.

`ParticleStar` tests the deuterium-burning model. It first evolves the model alone at constant \\(\dot{m} = 10^{-5}\\) and \\(10^{-4}\\,M\_\odot\\,{\rm yr}^{-1}\\), with the steps of the ORION2 driver, and compares radius, luminosity, \\(n\\), \\(m\_d\\) and phase with reference tracks computed by the ORION2 code (stored in `src/problems/ParticleStar/`; radius and luminosity agree within 0.75%). It then runs a \\(32^3\\) MHD simulation with self-gravity in which one Jeans-unstable cell forms a Star particle, and checks mass conservation in the formation step (to \\(10^{-14}\\)) and during accretion (to \\(10^{-13}\\)), and that the model has started and gives \\(1.5 \le n \le 3\\) and a positive luminosity.
