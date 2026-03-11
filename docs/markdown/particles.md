# Particles, star formation and feedback

## StochasticStellarPop Particle Type

StochasticStellarPop particles carry the following real-valued attributes:

- **Mass**: Current particle mass ($M_{\star}$)
- **Velocity**: Three-dimensional velocity vector $(v_x, v_y, v_z)$
- **Birth time**: Simulation time when the particle formed
- **Death time**: When the particle explodes as a supernova
- **Mass at birth**: Initial mass before mass loss
- **Luminosity**: Radiation luminosity (if radiation is enabled)

Each particle also stores an integer **evolution stage** that tracks its lifecycle:

- `SNProgenitor`: High-mass star ($M > 8 M_{\odot}$) that will explode as a supernova
- `SNRemnant`: Compact remnant left after supernova explosion
- `LowMassStar`: Low-mass star that will not explode (not used in the current star formation implementation)
- `LowMassComposite`: Composite particle representing a population of low-mass stars

## Sink Particle Type

Sink particles carry the following real-valued attributes:

- **Mass**: Current particle mass ($M$)
- **Velocity**: Three-dimensional velocity vector $(v_x, v_y, v_z)$

Unlike StochasticStellarPop particles, sink particles have no evolution stage or birth/death metadata. They represent unresolved gravitationally-collapsed objects that accrete mass from the surrounding gas.

## Sink Particle Formation

### Overview

The sink formation module creates sink particles from gas cells that violate the Jeans criterion (Truelove et al. 1997). The algorithm is operator-split from the hydrodynamics and runs once per hydro timestep at the finest AMR level.

### Formation criteria

A sink particle is created in a cell if **all** of the following conditions are met:

1. **Jeans instability**: The cell density exceeds the local Jeans density,

$$
\rho > \rho_J = J^2 \frac{\pi c_{\text{eff}}^2}{G \Delta x^2},
$$

where $J = 0.25$ is the Jeans number and $c_{\text{eff}} = c_s \sqrt{1 + 0.74 / \beta}$ is the effective sound speed accounting for magnetic pressure support ($\beta = P_{\text{thermal}} / P_{\text{magnetic}}$). For non-MHD simulations, $\beta \to \infty$ and $c_{\text{eff}} = c_s$.

2. **Not in an existing accretion zone**: The cell is not already being accreted by an existing sink particle (i.e., the accretion rate in the cell is zero).

3. **Local density maximum**: The cell has the highest density among all cells within a sphere of radius $3 \Delta x$. This prevents multiple sink particles from forming in close proximity.

### Mass and momentum assignment

When a sink particle forms, it receives the excess mass above the Jeans density:

$$
m_{\text{particle}} = (\rho - \rho_J) \, \Delta x^3
$$

The particle inherits the gas velocity of the parent cell. The cell density is then set to $\rho_J$, and all conserved quantities (momentum, energy, internal energy) are scaled by the factor $\rho_J / \rho$.

### Practical considerations

- The local maximum check uses a strict inequality ($\rho_{\text{neighbor}} > \rho_{\text{cell}}$), so ties do not prevent formation. In practice, exact density ties only occur in artificial initial conditions.
- Sink formation is applied after sink accretion in each timestep, so newly formed particles do not overlap with existing accretion zones.

## Sink Particle Accretion

### Overview

Sink particle accretion follows the Bondi-Hoyle-Lyttleton prescription of Krumholz et al. (2004). Gas is removed from cells surrounding each sink particle and added to the particle's mass, with the accretion rate set by the local gas properties and the particle mass.

### Accretion rate

The Bondi-Hoyle radius is

$$
r_{\text{BH}} = \frac{G M}{v_\infty^2 + c_{f,\infty}^2},
$$

where $M$ is the particle mass, $v_\infty$ is the gas velocity relative to the particle (mass-weighted mean over the accretion zone), and $c_{f,\infty}$ is the fast magnetosonic speed (mass-weighted mean over the accretion zone). For MHD, $c_f^2 = c_s^2 (1 + 2/\beta)$; for non-MHD, $c_f = c_s$.

The accretion rate is

$$
\dot{M} = 4 \pi \rho_\infty r_{\text{BH}}^2 \sqrt{v_\infty^2 + \lambda^2 c_{f,\infty}^2},
$$

where $\rho_\infty$ is the mean gas density in the accretion zone and $\lambda = e^{3/2}/4$.

### Accretion kernel

The accretion zone is a sphere of radius $r_{\text{acc}} = 3 \Delta x$ centered on the particle. Within this zone, each cell receives a Gaussian weight:

$$
w = \exp\left(-r^2 / r_K^2\right),
$$

where $r$ is the distance from the particle to the cell centre. The kernel radius $r_K$ is resolution-adaptive:

$$
r_K = \begin{cases}
\Delta x / 4 & r_{\text{BH}} < \Delta x / 4, \\
r_{\text{BH}} & \Delta x / 4 \leq r_{\text{BH}} \leq r_{\text{acc}} / 2, \\
r_{\text{acc}} / 2 & r_{\text{BH}} > r_{\text{acc}} / 2.
\end{cases}
$$

The accretion rate deposited in each cell is $\dot{M} \, w_i / \sum_i w_i$. Optionally, a uniform kernel ($w = 1$ for all cells in the $(7 \Delta x)^3$ stencil) can be used for testing by setting `particles.sink_particle_use_uniform_kernel = 1`.

### Accretion limiters

Two corrections are applied to the per-cell accretion rate, in the following order:

1. **Mass removal cap**: No more than 25% of a cell's mass may be removed in a single timestep (Krumholz et al. 2004). This prevents artificial sound waves from being launched by rapid density changes.

2. **Jeans density floor**: If the post-accretion cell density would still exceed the Jeans density $\rho_J$, the accretion rate is increased so that the final density equals $\rho_J$. This is safe because such cells are at the centre of highly supersonic convergence and are causally disconnected from their surroundings.

### Momentum accretion

When gas is accreted, the particle gains both mass and momentum. The new particle velocity is computed from momentum conservation:

$$
\vec{v}_{\text{new}} = \frac{m_{\text{old}} \vec{v}_{\text{old}} + \Delta m \, \vec{v}_{\text{gas}}}{m_{\text{old}} + \Delta m},
$$

where $\Delta m$ is the total accreted mass and $\vec{v}_{\text{gas}}$ is the mass-weighted gas velocity over the accreted cells. The gas state is then updated by scaling all conserved quantities by $(1 + \dot{M}_{\text{cell}} \Delta t / (\rho V))$, which preserves the gas velocity field.

### Galilean invariance

The accretion rate is computed using gas velocities in the particle frame ($\vec{v}_\infty = \vec{v}_{\text{gas}} - \vec{v}_{\text{particle}}$), ensuring Galilean invariance.

### Runtime Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles.sink_particle_use_uniform_kernel` | Boolean | `0` | Use uniform accretion kernel (for testing) |

### Examples

#### ParticleSinkFormation Test

The `ParticleSinkFormation` test validates combined sink particle formation and accretion. The test checks:

1. Exactly one star forms in the first timestep.
2. Mass is conserved to machine precision during sink formation.
3. Mass is conserved to machine precision during sink accretion.

#### ParticleSink Test

The `ParticleSink` test validates Bondi-Hoyle accretion and Galilean invariance. It runs in three phases:

1. **Base simulation**: Runs with zero boost velocity and validates the density profile against an analytical solution.
2. **Boosted simulation**: Runs with a boost velocity of $10^8$ cm/s and verifies that the density profile matches the analytical solution, demonstrating Galilean invariance.
3. **Multi-timestep evolution**: Continues the boosted simulation for additional timesteps and validates total mass conservation to machine precision.

## Star Formation

### Overview

The star formation module adds star particles through a stochastic prescription that plugs into the Stochastic Stellar Population specialisation.

- Runs once per hydro timestep and evaluates each cell independently.
- Always spawns a `LowMassComposite` particle when star formation is triggered.
- Adds high-mass particles ($> 8~M_{\odot}$) probabilistically so the Chabrier (2005) initial mass function (IMF) is satisfied in expectation.

### Jeans instability filter

Eligible cells are first identified through a Jeans-length check before any stochastic sampling occurs.

- Compute the Jeans density $\rho_J = J^2 \pi c_{\text{eff}}^2 / (G \Delta x^2)$, where $J = 0.25$ is the Jeans number and $c_{\text{eff}} = c_s \sqrt{1 + 0.74 / \beta}$ is the effective sound speed, accounting for thermal pressure and magnetic pressure with $\beta = P_{\text{thermal}} / P_{\text{magnetic}}$.
- Mark the cell as eligible when $\rho > \rho_J$.
- Only eligible cells continue to the sampling steps below.

### Controlling the formation rate

Two efficiency parameters tune how aggressively eligible gas is converted into star particles during each hydro step.

- \(\epsilon_{ff}\): efficiency per free-fall time, defined through \(\epsilon_{ff} = (\dot M_{\star} \, t_{ff}) / (M_{cell} \, \Delta t)\).
- \(\epsilon_{\star}\): fraction of the cell mass used when a particle is spawned.
- The target stellar mass for the step is \(\epsilon_{ff} M_{cell} (\Delta t / t_{ff})\).
- Bernoulli probability for spawning: \( P = \frac{\epsilon_{ff}}{\epsilon_{\star}} \frac{\Delta t}{t_{ff}} \).
- The expectation value \(\langle M_{\star} \rangle = P \epsilon_{\star} M_{cell}\) matches the target mass provided \(\Delta t < t_{ff}\); the CFL condition typically enforces that inequality.

### Sampling the stellar population

Once a cell passes the filter and the Bernoulli draw succeeds, we construct the composite stellar population represented by the spawned particles.

- Every accepted draw creates one low-mass particle that represents all stars with \(M < 8 M_{\odot}\).
- High-mass stars follow the Chabrier (2005) IMF: log-normal below \(1 M_{\odot}\) and a slope of \(2.35\) above it.
- Pre-computed IMF integrals provide the mass fraction \(f_{\star,high}\) and mean mass \(\langle m \rangle_{\star,high}\) of the high-mass component.
- The expected number of massive stars is \(f_{\star,high} \epsilon_{\star} M_{cell} / \langle m \rangle_{\star,high}\); a Poisson variate with that mean sets the actual count.
- Each massive star draws its mass from the high-mass end of the IMF, while the low-mass particle retains the remaining fraction \(1 - f_{\star,high}\) of the spawned mass.
- If the Poisson draw returns zero, only the low-mass particle is inserted and it inherits the local gas velocity.

### Assigning particle velocities

Sampled star particles receive velocities that combine the local bulk flow with an isotropic runaway kick.

- Each massive star draws a speed from a power-law distribution \(p(v) \propto v^{-1.8}\) truncated between 3 and 385 km s\(^{-1}\), then converts it to cgs units.
- A random direction is chosen by sampling \(\cos \theta\) uniformly in \([-1, 1]\) and \(\phi\) in \([0, 2\pi)\); the resulting kick is added to the gas velocity of the parent cell.
- The total momentum of all massive stars is accumulated, and the low-mass composite particle receives the opposite momentum so that the cell-level particle system conserves momentum.

### Practical considerations

A few implementation notes help interpret corner cases and limitations of the current recipe.

- Star formation is operator-split from the hydrodynamics. When \(t_{ff}\) is unresolved (\(\Delta t \gtrsim t_{ff}\)), the true star formation rate is not captured, and this scheme provides one possible approximation; no explicit limiter is enforced beyond the CFL-controlled hydro step.
- All spawned particles are inserted at the cell centre. Other physics modules are responsible for any subsequent repositioning or feedback coupling.


## Supernova Feedback

### Overview

Quokka includes a particle framework for modeling stellar populations and their feedback effects on the interstellar medium (ISM). The primary feedback mechanism is supernova (SN) explosions, which inject energy and momentum into the surrounding gas. The SN module implements several physically-motivated schemes that adapt to the local resolution and density conditions.

### Supernova Feedback

#### Physical Parameters

When a progenitor star reaches its death time, it explodes as a Type II supernova with the following canonical parameters:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Blast energy | $E_{\text{blast}}$ | $10^{51}$ erg | Thermal energy injected into the ISM |
| Ejecta mass | $m_{\text{ej}}$ | $10 M_{\odot}$ | Mass expelled during explosion |
| Terminal momentum | $p_{\text{snr},0}$ | $2.8 \times 10^5 M_{\odot} \, \text{km s}^{-1}$ | Asymptotic momentum of the SNR (configurable via `particles.SN_p_term_Msunkmps`) |
| Remnant mass | $m_{\text{dead}}$ | $\geq 1.4 M_{\odot}$ | Mass of the compact remnant |
| Kinetic energy | $E_{\text{kin}}$ | $0.5 m_{\text{ej}} v_{\text{star}}^2$ | Kinetic energy of the ejecta |

The terminal momentum is density-dependent and scales as:

$$
p_{\text{snr}} = p_{\text{snr},0} \, n_{\text{H}}^{-0.17}
$$

where $n_{\text{H}}$ is the ambient hydrogen number density averaged over the deposition kernel.

#### Deposition Kernel

SN feedback is deposited into a $(2 \times 3 + 1)^3 = 343$ cell cubic stencil centered on the particle's location. The kernel weights are pre-computed to approximate a spherical distribution with radius $r = 3 \Delta x$, where $\Delta x$ is the cell width, around the host cell center. The kernel weights $W_{ijk}$ are normalized such that $\sum_{i,j,k} W_{ijk} = 1$.

#### Resolution-Adaptive Schemes

The feedback implementation uses the **resolution factor** $R_M$ to determine whether the Sedov-Taylor phase is resolved:

$$
R_M = \frac{M_{\text{SNR}}}{M_{\text{sf}}}
$$

where:
- $M_{\text{SNR}} = M_{\text{gas}} + m_{\text{ej}}$ is the total SNR mass (gas in stencil plus ejecta)
- $M_{\text{sf}} = 1679 M_{\odot} \, n_{\text{H}}^{-0.26} \left(p_{\text{snr},0} / p_{\text{snr},0}^{\text{canonical}}\right)^2$ is the shell-formation mass, scaled so that the kinetic energy $p_{\text{snr}}^2 / (2 M_{\text{sf}})$ is invariant when $p_{\text{snr},0}$ is changed

When $R_M < 1$, the Sedov-Taylor phase is resolved and the blast wave dynamics can be captured. When $R_M \geq 1$, the resolution is insufficient and the code transitions to momentum-dominated feedback following Kim & Ostriker (2017).

##### Available Schemes

Quokka provides four SN feedback schemes controlled by `particles.SN_scheme`:

1.  **`SN_thermal_only`**: Pure thermal energy injection

    - Deposits $E_{\text{blast}} + E_{\text{kin}}$ into gas total energy
    - Deposits ejecta mass $m_{\text{ej}}$ and momentum $m_{\text{ej}} \vec{v}_{\text{ej}}$ into gas momentum. No further momentum injection
    - Simplest scheme, appropriate when Sedov-Taylor phase is well-resolved. Should only be used for testing.

2.  **Thermal and momentum schemes**: The following schemes include resolution-dependent momentum injection. In these schemes, an energy of $E_{\text{blast}}$ + $E_{\text{kin}}$ is injected into the gas total energy, and a fraction $f$ of the asymptotic momentum $p_{\text{snr}}$ is injected as momentum: $p_{\text{inject}} = f \, p_{\text{snr}}$. The difference between the injected total energy and kinetic energy remains as thermal energy. Note that there is always some amount of thermal energy injected, even when $f = 1$. The momentum fraction $f$ is determined by the resolution factor $R_M$ as follows:

    a. **`SN_thermal_or_thermal_momentum`** (default):

       - When $R_M < 0.027$: Pure thermal (like `SN_thermal_only`) with $f = 0$
       - When $0.027 \leq R_M < 1$: Partial momentum injection with $f = 0.529 \sqrt{R_M}$. Based on Kim & Ostriker (2017) calibration, $f^2 = 0.28 R_M$ means that 28% of the total energy is kinetic energy in this case.
       - When $R_M \geq 1$: Full momentum injection with $f = 1$

    b. **`SN_thermal_kinetic_or_thermal_momentum`**:

       - When $R_M < 0.027$: Pure kinetic injection with $f = \sqrt{2 R_M}$
       - When $0.027 \leq R_M < 1$: Partial momentum injection with $f = 0.529 \sqrt{R_M}$
       - When $R_M \geq 1$: Full momentum injection with $f = 1$

    c. **`SN_pure_kinetic_or_thermal_momentum`**:

       - Always injects full terminal momentum regardless of resolution with $f = 1$

#### Momentum Deposition

For schemes with momentum injection, the following momentum deposition procedure guarantees Galilean invariance for a single SN event:

1. **Compute COM velocity** of the SNR (gas + ejecta):

$$
\vec{v}_{\text{COM}} = \frac{\vec{P}_{\text{gas}} + m_{\text{ej}} \vec{v}_{\text{ej}}}{M_{\text{gas}} + m_{\text{ej}}}
$$

where $\vec{P}_{\text{gas}}$ is the total momentum of gas in the stencil (kernel-weighted).

2. **Deposit momentum** such that each cell receives:

$$
\Delta \vec{p}_{ijk} = \left[\rho_{\text{new}} \vec{v}_{\text{COM}} - \vec{p}_{\text{old}}\right] + \vec{p}_{\text{radial}}
$$

where $\vec{p}_{\text{radial}} = f \, p_{\text{snr}} \, W_{ijk} \, \hat{\mathbf{r}}_{ijk}$ is the radial momentum component.

3. **Energy deposition** includes a cross term:

$$
\Delta E_{ijk} = \left(E_{\text{blast}} + E_{\text{kin}}\right) W_{ijk} + \vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial}}
$$

The cross term $\vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial}}$ accounts for the kinetic energy change from the radial expansion, ensuring Galilean invariance. This term sums to zero over all cells in the stencil, (momentum is conserved), ensuring energy conservation.

**Background smoothing**. To ensure the cross term cancels when summing over all cells in the stencil, the velocity field (but not the density) must be smoothed such that $\sum_i \vec{v}_{\text{COM}} \cdot \vec{p}_{\text{radial},i} = \vec{v}_{\text{COM}} \cdot \sum_i \vec{p}_{\text{radial},i} = 0$. In our tests, this smoothing also dramatically reduces peak velocities in SNR. In a shearing/turbulent background, this artificially homogenizes the gas velocity and changes kinetic energy unrelated to the SN. We provide a parameter `particles.SN_smooth_gas_velocity=0` to turn off this smoothing: $\Delta \vec{p}_{ijk} = m_{\text{ej}} \vec{v}_{\text{ej}} / V + \vec{p}_{\text{radial}}$ and $\Delta E_{ijk} = \left(E_{\text{blast}} + E_{\text{kin}}\right) W_{ijk} + \vec{v}_{i} \cdot \vec{p}_{\text{radial}}$. In this case, the cross term sums to a non-zero value: $\sum_i \vec{v}_i \cdot \vec{p}_{\text{radial},i} = (\vec{v}_{\text{COM}} + \delta v_i) \cdot \sum_i \vec{p}_{\text{radial},i} = \sum_i \delta v_i \cdot \vec{p}_{\text{radial},i}$, where $\delta v_i$ is the velocity of cell $i$ relative to the COM frame.

### Runtime Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `particles.SN_scheme` | String | `SN_thermal_or_thermal_momentum` | Feedback scheme (see above) |
| `particles.SN_p_term_Msunkmps` | Float | `2.8e5` | Terminal momentum $p_{\text{snr},0}$ in units of $M_\odot\,\mathrm{km\,s}^{-1}$. The shell-formation mass $M_\mathrm{sf}$ is scaled as $(p/p_\mathrm{canonical})^2$ to preserve the kinetic energy $p^2/(2M_\mathrm{sf})$. |
| `particles.disable_SN_feedback` | Boolean | `0` | Disable SN feedback entirely |
| `particles.verbose` | Integer | `0` | Verbosity level for particle diagnostics |
| `particles.stellar_velocity_limit` | Float | $10^8$ cm/s | Maximum allowed stellar velocity (aborts if exceeded) |
| `particles.SN_smooth_gas_velocity` | Integer | `1` | Smooth gas velocity in the stencil to enforce energy conservation |

### 	Implementation Notes

#### Operator Splitting

SN feedback is operator-split from the hydrodynamics and applied after each timestep:

1. Hydrodynamics advances the gas state by $\Delta t$
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

The terminal momentum $p_{\text{snr}}$ represents the asymptotic momentum that a supernova remnant reaches in the pressure-confined snowplow phase, after most of the kinetic energy has been radiated away. This value (Kim & Ostriker 2017) is calibrated from high-resolution simulations and depends weakly on the ambient density.

The $R_M$ factor effectively measures whether the simulation has sufficient resolution and density to capture the Sedov-Taylor expansion. When $R_M \ll 1$, the blast wave will expand through many cells before reaching the shell-formation radius, allowing the hydrodynamics to naturally capture the momentum buildup. When $R_M \sim 1$ or larger, the explosion is under-resolved and the code preemptively injects the expected terminal momentum.

### Examples

#### Basic SN Test

The `SN` problem provides a canonical test with one SN progenitor in a uniform medium:

```
particles.SN_scheme = SN_thermal_or_thermal_momentum
problem.boost_vel_x = 1.0e8 # boost velocity in x direction
```

This test is run three times, with ambient density of $10, 1.0$, and $0.1 ~\mathrm{cm}^{-3}$, corresponding to a shell-formation radius of 7, 22, and 70 pc, respectively. The spatial resolution is 5 pc. The default `SN_thermal_or_thermal_momentum` scheme is used. The three ambient densities implies $R_M \approx (3 \Delta x / R_{\text{sf}})^3 = 10, 0.3$, and $0.01$, respectively, placing the scheme in the under-resolved, partially resolved, and well-resolved regimes, respectively.

In each of the three tests, the simulation is run twice, one initially at rest and one with an initial boost velocity of $200$ km/s, to demonstrate Galilean invariance. The temperature and velocity profiles are identical in the boosted and rest frames (to some tolerance).

#### Random Blast Test

The `RandomBlast` problem provides a testbed for multiple SN explosions with various ambient conditions and bulk flows.
